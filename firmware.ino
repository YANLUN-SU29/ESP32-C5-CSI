#include <WiFi.h>
#include <WiFiClient.h>
#include "esp_wifi.h"
#include "ping/ping_sock.h"
#include "lwip/ip_addr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// ===== 目標晶片：ESP32-C5 =====
// 需要 arduino-esp32 3.3.0 以上（ESP-IDF 5.5），開發板選「ESP32C5 Dev Module」。
// C5 的 CSI 設定結構、rx_ctrl 欄位與緩衝區佈局都和 S3 不同，S3 版不能直接燒。
#if !CONFIG_IDF_TARGET_ESP32C5
#error "此韌體僅適用 ESP32-C5（S3 版請用 ESP32-CSI 專案的 firmware.ino）"
#endif


const char* ssid = "WIFI CSI";
const char* password = "Louis229";


// ===== 頻段選擇（C5 為雙頻，路由器 TP-Link AC750 也是雙頻）=====
// 0 = 只連 2.4 GHz，1 = 只連 5 GHz
//
// 必須固定單一頻段：設成自動的話，重連時可能換到另一個頻段，
// 波長、通道、子載波間距全變，前後錄的資料就不能比。
// 5 GHz 波長約 5.7cm（2.4 GHz 為 12.5cm），同樣的胸腔位移造成的相位變化約大一倍，
// 且鄰居干擾較少，故預設 5 GHz。
// 注意：路由器兩個頻段若用同一個 SSID，請在路由器端拆開或確認此設定生效。
#define WIFI_USE_5G 1


// ===== 觸發方式切換 =====
// 1 = ICMP ping：路由器對 echo request 是即時回應，不受 TCP delayed ACK 影響
// 0 = 原本的 TCP 敲門
//
// 改用 ping 的原因：實測 TCP 敲門雖每 10ms 送一次（預期 ~100 CSI/s），
// 實際只拿到 59/s，且封包成群到達、長間隔穩定在 47ms（變異係數僅 0.13），
// 指向路由器的 TCP delayed ACK 計時器。ICMP echo 不走該路徑。
// 註：ICMP 路徑尚未實機驗證，故預設維持已驗證的 TCP 模式。
//     要試 ping 請改為 1，並比對 monitor.py 結束統計的平均 fps。
//     （以上實測數據來自 S3 + N300 路由器，換成 C5 + TP-Link 後需重新量測。）
#define USE_ICMP_PING 0


WiFiClient client;
IPAddress gateway; // 用來自動儲存路由器的 IP
bool tcp_connected = false; // TCP 連線狀態旗標
esp_ping_handle_t ping_handle = NULL;
uint8_t ap_bssid[6] = {0}; // 只收路由器送來的封包，連線後填入


// ===== CSI 事件佇列 =====
// 原本在 CSI 回呼裡直接 Serial.println。該回呼跑在 Wi-Fi driver task 的 context，
// 阻塞式序列埠寫入（一行約 490 bytes、@921600 baud 需 ~5ms）會卡住該任務，
// 導致 Wi-Fi 驅動不再把後續 CSI 事件送進來——而且這種「事件根本沒進到回呼」的
// 損失在 PC 端完全偵測不到，連計數器都測不出來。
//
// 改成：回呼只負責把資料丟進佇列（非阻塞），由 loop() 排空並輸出。
// 佇列滿代表序列埠追不上，此時明確計數為 queue_full_drop——損失變成可量測的數字。
//
// C5 的 len 依封包型別而定（ESP-IDF 文件）：HT20/VHT20 為 114 bytes（57 條），
// STBC 228、HT40 234、HE20 490。預期只會收到 114，但上限放寬到能容納 HE20，
// 讓非預期型別仍能送到 PC 端被統計出來，而不是在這裡被默默當成長度異常。
#define CSI_MAX_LEN   512
#define CSI_QUEUE_LEN 24    // 24 * ~524 bytes ≈ 12.6KB（C5 有 384KB SRAM，足夠）

typedef struct {
  uint32_t seq;       // 單調遞增序號，PC 端可據此偵測傳輸中遺失的行
  uint16_t len;
  int8_t   rssi;
  uint8_t  bb_format; // rx_ctrl.cur_bb_format：1=11a/g 2=HT 3=VHT 4~7=HE
  uint8_t  channel;
  uint8_t  second;    // 0=無次通道（20MHz），1=above，2=below
  uint8_t  first_word_invalid;
  int8_t   buf[CSI_MAX_LEN];
} csi_item_t;

QueueHandle_t csi_queue = NULL;

// 這些計數器由回呼（Wi-Fi task）寫入、loop 讀取，故標為 volatile
volatile uint32_t csi_seq          = 0;  // 進入回呼且通過檢查的事件總數
volatile uint32_t queue_full_drop  = 0;  // 因佇列滿而丟棄（序列埠追不上）
volatile uint32_t oversize_drop    = 0;  // 因長度異常而丟棄
volatile uint32_t invalid_drop     = 0;  // 硬體標記 CSI 無效（rx_channel_estimate_info_vld=0）


// CSI 回呼：只做最小工作量，絕不阻塞
void _wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  // 只要路由器送來的封包：其他裝置的封包走的是不同路徑，
  // 混進來會讓同一個子載波的時間序列在兩條通道之間跳動。
  // 這類封包是刻意過濾，不算丟失，故不計數也不佔序號。
  if (memcmp(data->mac, ap_bssid, 6) != 0) {
    return;
  }

  // C5 的硬體會自報這筆通道估計是否有效（ESP-IDF 文件：為 0 時 CSI 無效）
  if (!data->rx_ctrl.rx_channel_estimate_info_vld) {
    invalid_drop++;
    return;
  }

  uint16_t len = data->len;

  // 長度異常直接丟棄，保護晶片不當機
  if (len == 0 || len > CSI_MAX_LEN) {
    oversize_drop++;
    return;
  }

  csi_item_t item;
  item.seq                = ++csi_seq;
  item.len                = len;
  item.rssi               = data->rx_ctrl.rssi;
  item.bb_format          = data->rx_ctrl.cur_bb_format;
  item.channel            = data->rx_ctrl.channel;
  item.second             = data->rx_ctrl.second;
  item.first_word_invalid = data->first_word_invalid;
  memcpy(item.buf, data->buf, len);

  // 逾時 0：佇列滿就立刻放棄並計數，絕不在此等待
  if (xQueueSend(csi_queue, &item, 0) != pdTRUE) {
    queue_full_drop++;
  }
}


// 由 loop() 呼叫，把佇列內的 CSI 格式化後輸出
// 行格式：CSI_V4,<seq>,<len>,<rssi>,<bb_format>,<dropped>,<channel>,<second>,<fwi>,<csi...>
//   seq       = 事件序號，PC 端比對跳號即知傳輸中遺失了幾行
//   dropped   = ESP32 端累計丟棄數（佇列滿 + 長度異常 + 硬體標記無效）
//   bb_format = 取代 S3 的 sig_mode（C5 的 rx_ctrl 沒有 sig_mode）
//   fwi       = first_word_invalid，1 表示前 4 bytes（前兩條子載波）因硬體限制無效
// 回傳是否有輸出任何資料，供 loop 決定要不要讓出 CPU
bool drain_csi_queue() {
  static char line_buf[4096];
  csi_item_t item;
  bool any = false;

  while (xQueueReceive(csi_queue, &item, 0) == pdTRUE) {
    any = true;
    uint32_t dropped = queue_full_drop + oversize_drop + invalid_drop;
    int pos = 0;
    pos += sprintf(line_buf + pos, "CSI_V4,%lu,%u,%d,%u,%lu,%u,%u,%u",
                   (unsigned long)item.seq, (unsigned)item.len,
                   (int)item.rssi, (unsigned)item.bb_format,
                   (unsigned long)dropped, (unsigned)item.channel,
                   (unsigned)item.second, (unsigned)item.first_word_invalid);
    for (int i = 0; i < item.len; i++) {
      pos += sprintf(line_buf + pos, ",%d", item.buf[i]);
    }
    Serial.println(line_buf);
  }
  return any;
}


// ===== ICMP ping 觸發 =====
// 回呼留空即可：我們不在意 ping 的統計結果，只要它持續產生單播往返，
// 讓路由器的 echo reply 觸發 CSI 回呼。
static void on_ping_success(esp_ping_handle_t hdl, void *args) {}
static void on_ping_timeout(esp_ping_handle_t hdl, void *args) {}
static void on_ping_end(esp_ping_handle_t hdl, void *args) {}

void start_ping_session() {
  esp_ping_config_t cfg = ESP_PING_DEFAULT_CONFIG();

  ip_addr_t target;
  IP_ADDR4(&target, gateway[0], gateway[1], gateway[2], gateway[3]);
  cfg.target_addr = target;
  cfg.count = ESP_PING_COUNT_INFINITE; // 不停止
  cfg.interval_ms = 10;                // 與原本 TCP 敲門的 delay(10) 對齊，便於比較
  cfg.timeout_ms = 200;
  cfg.data_size = 32;                  // 小封包即可，目的只是觸發往返

  esp_ping_callbacks_t cbs;
  cbs.cb_args = NULL;
  cbs.on_ping_success = on_ping_success;
  cbs.on_ping_timeout = on_ping_timeout;
  cbs.on_ping_end = on_ping_end;

  if (esp_ping_new_session(&cfg, &cbs, &ping_handle) == ESP_OK) {
    esp_ping_start(ping_handle);
    Serial.printf("[Trigger] ICMP ping -> %s, interval %ums\n",
                  gateway.toString().c_str(), cfg.interval_ms);
  } else {
    Serial.println("[Error] 無法建立 ping session，請改用 TCP 模式 (USE_ICMP_PING 0)");
  }
}


void setup() {
  Serial.begin(921600); // 高速鮑率，大幅提升傳輸吞吐量
  Serial.println("\n[ESP32-C5] CSI Radar (TCP Keep-Alive / 921600 baud)");


  // WiFi.mode() 會啟動 Wi-Fi，頻段與頻寬都必須在啟動後、連線前設定
  WiFi.mode(WIFI_STA);

#if WIFI_USE_5G
  WiFi.setBandMode(WIFI_BAND_MODE_5G_ONLY);
#else
  WiFi.setBandMode(WIFI_BAND_MODE_2G_ONLY);
#endif

  // 【鎖定 20 MHz】只有 20 MHz 封包的 CSI 佈局固定為 114 bytes（57 條子載波）。
  // TP-Link AC750 的 5 GHz 預設頻寬為自動（最高 80 MHz），若讓它送 HT40，
  // len 會變 234，同一陣列索引對應到不同的實體子載波，PC 端只能整幀丟棄。
  // 在 STA 端宣告只支援 20 MHz，路由器就只會用 20 MHz 跟我們傳。
  wifi_bandwidths_t bw = { .ghz_2g = WIFI_BW_HT20, .ghz_5g = WIFI_BW_HT20 };
  esp_err_t bw_err = esp_wifi_set_bandwidths(WIFI_IF_STA, &bw);
  if (bw_err != ESP_OK) {
    Serial.printf("[Warn] 無法鎖定 20MHz 頻寬：%s（請在路由器端把頻寬設為 20MHz）\n",
                  esp_err_to_name(bw_err));
  }

  WiFi.begin(ssid, password);

  // 【關閉 Wi-Fi 省電】STA 模式預設會開啟 modem sleep，在信標間隔之間關掉射頻，
  // 封包被路由器緩衝後才一次灌下來，造成「一陣密集 + 一段靜默」的不均勻取樣。
  // 實測舊韌體的樣本間隔呈雙峰分布（約 8.5ms 與 50ms，中間完全空白），
  // 而 Welch PSD 與 Butterworth 濾波都假設均勻取樣，故必須關閉。
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[OK] Wi-Fi Connected");
  Serial.println("[Power] Wi-Fi modem sleep disabled (WIFI_PS_NONE)");
  Serial.printf("[Link] Channel %ld (%s), RSSI %d dBm, BSSID %s\n",
                (long)WiFi.channel(), WiFi.channel() > 14 ? "5GHz" : "2.4GHz",
                WiFi.RSSI(), WiFi.BSSIDstr().c_str());

  // 記下路由器的 BSSID，CSI 回呼據此過濾
  memcpy(ap_bssid, WiFi.BSSID(), 6);


  // 自動取得 TP-Link 路由器的 IP
  gateway = WiFi.gatewayIP();
  Serial.print("[Target] Gateway IP: ");
  Serial.println(gateway);


  // C5 的 CSI 設定與 S3 完全不同（S3 是 lltf_en/htltf_en...，C5 是逐型別開關）。
  // 每個封包只會帶一種 LTF，不像 S3 會把 LLTF 與 HT-LTF 串在一起。
  //
  // - 只開 ht20 與 vht：路由器在 5 GHz 用 11ac 傳資料時是 VHT20，2.4 GHz 則是 HT20，
  //   兩者佈局相同（ESP-IDF 文件：VHT 與 HT 的子載波索引、總長度一致）。
  // - legacy 關閉：11a/g 封包（信標等）只有 LLTF，長度 106、佈局不同，混進來無益。
  // - force_lltf 關閉：開啟後改為只取 LLTF（52 條可用），比 HT-LTF 的 56 條少，
  //   且官方範例對其位元格式的處理與文件描述不一致，暫不採用。
  // - ht40 關閉：上面已鎖 20MHz，這裡再擋一次。
  // - HE（su/mu/dcm/beamformed）關閉：AC750 是 Wi-Fi 5，不會有 HE 封包。
  wifi_csi_config_t csi_config = {
      .enable                   = true,
      .acquire_csi_legacy       = false,
      .acquire_csi_force_lltf   = false,
      .acquire_csi_ht20         = true,
      .acquire_csi_ht40         = false,
      .acquire_csi_vht          = true,
      .acquire_csi_su           = false,
      .acquire_csi_mu           = false,
      .acquire_csi_dcm          = false,
      .acquire_csi_beamformed   = false,
      .acquire_csi_he_stbc_mode = 2,
      .val_scale_cfg            = 0,
      .dump_ack_en              = false,
      .lltf_bit_mode            = 0,
      .reserved                 = 0,
  };
  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));

  // 佇列必須在註冊回呼之前建好，否則第一個事件就會寫到 NULL
  csi_queue = xQueueCreate(CSI_QUEUE_LEN, sizeof(csi_item_t));
  if (csi_queue == NULL) {
    Serial.println("[Error] CSI 佇列建立失敗（記憶體不足）");
    while (true) delay(1000);
  }
  Serial.printf("[Queue] CSI 佇列 %d 筆 x %u bytes = %u bytes\n",
                CSI_QUEUE_LEN, (unsigned)sizeof(csi_item_t),
                (unsigned)(CSI_QUEUE_LEN * sizeof(csi_item_t)));

  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&_wifi_csi_cb, NULL));

  // 啟動 CSI 底層引擎（放在設定與回呼都就緒之後）
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));

#if USE_ICMP_PING
  start_ping_session();
  Serial.println("[Start] CSI capture active, ICMP ping mode...");
#else
  Serial.println("[Start] CSI capture active, TCP keep-alive mode...");
#endif
}


void loop() {
  // 每輪先盡量排空 CSI 佇列。這是 loop 最重要的工作——排得不夠快佇列就會滿，
  // 進而累加 queue_full_drop。原本的 delay(10) 會讓 loop 有 10ms 完全不排空，
  // 故改用 millis() 排程敲門，不再阻塞。
  bool drained = drain_csi_queue();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] 連線中斷，重新連線...");
    WiFi.reconnect();
    return;
  }

#if !USE_ICMP_PING
  // TCP 常駐連線：只在斷線時才重新建立
  // 第三個參數是連線逾時 (ms)。預設逾時可長達數秒，期間 loop() 被卡住、
  // 敲門封包停發、CSI 就斷流——實測 201 秒錄製出現 9 次中斷（最長 613ms）。
  // 壓成 500ms 可把單次卡頓上限鎖住。
  // （TP-Link 路由器的管理頁面同樣在 port 80，敲門目標不必改。）
  if (!client.connected()) {
    if (client.connect(gateway, 80, 500)) {
      // 關閉 TCP 緩衝延遲，封包隨發隨至，減少卡頓
      client.setNoDelay(true);
      tcp_connected = true;
    } else {
      tcp_connected = false;
    }
  }

  // 每 10ms 送一個最小封包保活，觸發路由器回傳單播封包
  static uint32_t last_knock = 0;
  if (millis() - last_knock >= 10) {
    last_knock = millis();
    if (tcp_connected && client.connected()) {
      client.print(" \r\n");
    } else {
      tcp_connected = false;
    }
  }
#endif

  // 只在「這輪沒東西可輸出」時才讓出 CPU。
  // 原本無條件 vTaskDelay(1) 會讓每一行都額外付 1ms：實測樣本間隔中位數
  // 由 5.00ms 變成 6.00ms、fps 由 85 掉到 76（約 -11%）。
  // 忙碌時不必讓出，因為 Serial.write 在緩衝區滿時本身就會讓出。
  if (!drained) {
    vTaskDelay(1);
  }
}
