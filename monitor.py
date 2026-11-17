import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import datetime
import os
import sys
import time

# ----------------- [動態標籤輸入] -----------------
VALID_LABELS = ["Empty_Room", "Walking", "Waving"]

if len(sys.argv) > 1:
    ACTION_LABEL = sys.argv[1]
else:
    print("[選擇] 請選擇本次錄製的動作標籤：")
    for i, label in enumerate(VALID_LABELS):
        print(f"  [{i}] {label}")
    print(f"  或直接輸入自訂名稱")
    while True:
        choice = input("\n請輸入編號或名稱: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(VALID_LABELS):
            ACTION_LABEL = VALID_LABELS[int(choice)]
            break
        elif choice:
            ACTION_LABEL = choice
            break
        print("[錯誤] 輸入不能為空，請重新輸入。")

BAUD_RATE = 921600
WINDOW_SIZE = 100
GAP_WARN_THRESHOLD = 0.5  # 秒；相鄰兩筆樣本間隔超過此值視為訊號斷點

# ----------------- [子載波擷取佈局] -----------------
# ESP32-C5 與 S3 的佈局完全不同，S3 的 LLTF/HT-LTF 索引不能沿用：
#   - S3：一個封包同時帶 LLTF + HT-LTF (+ STBC)，各 64 條串在一起（256/384 bytes）
#   - C5：一個封包只帶一種 LTF。韌體只開 HT20/VHT20，固定為 114 bytes = 57 條
#
# C5 HT20/VHT20 的陣列順序（實測，2026-11-18）：單純依頻率由低到高
#   陣列索引  0~27 → 子載波 -28 ~ -1
#   陣列索引    28 → DC（恆為 0）
#   陣列索引 29~56 → 子載波  +1 ~ +28
#
# 這和 ESP-IDF 文件的表格（「0~28, -28~-1」）不同。舊版照文件寫，結果 DC 落在
# 我們標成 +28 的欄位而恆為 0。以 data/CSI_test 的相位驗證：相鄰子載波的相位應平滑
# 連續，照文件的排法在接合處誤差約 105°（等於完全對不上），照頻率排只有 4~10°；
# 振幅在索引 56→2 跳 37%（其實是頻帶兩端），在 27→29 只差 5.7%（隔著 DC 相鄰）。
#
# 只剔除 DC。索引 0、1（-28、-27）是 first_word_invalid 涵蓋的前 4 bytes，
# 但三筆空房間共 18.5 萬幀旗標全為 0，故保留；萬一某幀旗標為 1，沿用上一幀的值，
# 避免欄位時有時無、時間序列斷裂。
EXPECTED_LEN = 114                                        # bytes，其他長度整幀丟棄
DC_IDX = 28
FWI_IDX = (0, 1)                                          # first_word_invalid 涵蓋的陣列索引
SUBCARRIER_IDX = [k for k in range(EXPECTED_LEN // 2) if k != DC_IDX]   # 56 條（陣列索引）
SUBCARRIER_FREQ = [k - DC_IDX for k in SUBCARRIER_IDX]    # 對應的子載波編號 -28..-1, +1..+28
SUBCARRIERS = len(SUBCARRIER_IDX)

# rx_ctrl.cur_bb_format 的意義（esp_wifi_he_types.h）
BB_FORMAT_NAMES = {0: "11b", 1: "11a/g", 2: "HT", 3: "VHT", 4: "HE-SU",
                   5: "HE-MU", 6: "HE-ERSU", 7: "HE-TB", 11: "VHT-MU"}

# ----------------- [是否一併記錄相位] -----------------
# ESP32 量到的是複數 CSI，過去只存振幅 sqrt(I²+Q²)，把 atan2(Q,I) 丟掉了，
# 等於捨棄硬體提供的一半資訊。相位對微動偵測通常比振幅靈敏，
# 且這是唯一無法事後補救的一項——錄完才想加就得整批重錄，故預設開啟。
#
# 存的是「原始」相位：原始相位帶有 CFO/SFO 造成的隨機偏移，需跨子載波做
# 線性去趨勢校正才可用。校正屬分析階段的工作，採集階段不應破壞原始資料。
RECORD_PHASE = True

# 欄位名稱直接標出 LTF 種類與實體子載波編號（如 H-28、H5），讓資料自我描述。
# 開頭維持 H，batch_analyze 依此歸入 HT-LTF 組。
# 有相位時以 _A / _P 後綴區分，分析端才能只挑振幅欄（否則會把相位當成子載波混進 PCA）。
_BASE_NAMES = [f"H{f}" for f in SUBCARRIER_FREQ]
if RECORD_PHASE:
    COLUMN_NAMES = [f"{n}_A" for n in _BASE_NAMES] + [f"{n}_P" for n in _BASE_NAMES]
else:
    COLUMN_NAMES = list(_BASE_NAMES)

# ESP 端的封包抵達時間（微秒，韌體 CSI_V5 起提供），放在 Timestamp 之後。
# PC 端的 Timestamp 是在畫圖迴圈裡才蓋的，樣本一批一批到，不代表真正的取樣時刻。
ESP_TIME_COL = "EspTime_us"

print(f"\n[啟動] ESP32-C5 Wi-Fi 雷達 (CSV 錄製模式：{ACTION_LABEL} / 動態 COM 版)...")

# ----------------- [動態 COM Port 掃描] -----------------
def get_com_port():
    """挑選 ESP32 所在的序列埠。

    只考慮真正的 USB 裝置（有 VID/PID）：主機板內建的傳統序列埠
    （COM1，HWID 形如 ACPI\\PNP0501）沒有 VID/PID，且無法設定到 921600 baud，
    若誤鎖它會在開埠時拋出難以理解的 OSError(22, '參數錯誤')。
    """
    ports = list(serial.tools.list_ports.comports())
    usb_ports = [p for p in ports if p.vid is not None]

    if not usb_ports:
        print("[錯誤] 找不到任何 USB 序列裝置，ESP32 似乎沒有被系統辨識。")
        print("       請檢查：")
        print("         1. USB 線是否支援資料傳輸（純充電線不行）")
        print("         2. 是否已安裝 USB 轉序列埠驅動（ESP32-C5-DevKitC-1 的 UART 埠多為 CP2102N 或 CH343）")
        print("         3. 裝置管理員的「連接埠」或「其他裝置」有無帶驚嘆號的項目")
        if ports:
            print("       目前系統上只有這些非 USB 的序列埠（都不是 ESP32）：")
            for p in ports:
                print(f"         {p.device} - {p.description}")
        exit()

    if len(usb_ports) == 1:
        p = usb_ports[0]
        print(f"[自動] 鎖定唯一 USB 裝置：{p.device} ({p.description})")
        return p.device

    print("\n[掃描] 偵測到多個 USB 序列裝置，請選擇你的 ESP32：")
    for i, p in enumerate(usb_ports):
        print(f"  [{i}] {p.device} - {p.description}")
    while True:
        try:
            idx = int(input("\n請輸入設備號碼 (例如 0): ").strip())
            if 0 <= idx < len(usb_ports):
                print(f"[選擇] 已選擇：{usb_ports[idx].device}")
                return usb_ports[idx].device
            print("[錯誤] 號碼超出範圍，請重新輸入。")
        except ValueError:
            print("[錯誤] 請輸入有效的數字號碼。")

SERIAL_PORT = get_com_port()

# ----------------- [先連硬體，成功後才建立 CSV] -----------------
# 順序很重要：若先開檔再連線，連線失敗時會留下一堆只有標題列的空 CSV，
# 之後批次分析會踩到這些空檔。
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except Exception as e:
    print(f"[錯誤] 無法連線至 {SERIAL_PORT}: {e}")
    exit()
print(f"[連線] 已開啟 {SERIAL_PORT} @ {BAUD_RATE} baud")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

filename = os.path.join(DATA_DIR, f"CSI_{ACTION_LABEL}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_file = open(filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', ESP_TIME_COL] + COLUMN_NAMES)

print(f"[錄製] CSV 輸出路徑：{filename}")

# ----------------- [初始化矩陣] -----------------
data_matrix = np.zeros((SUBCARRIERS, WINDOW_SIZE))
record_count = 0
error_count = 0
gap_count = 0
layout_skip = 0       # 因長度不符預期佈局而丟棄的幀數
fwi_count = 0         # first_word_invalid=1 的幀數（該幀 -28、-27 沿用上一幀的值）
last_fwi_vals = {}    # 陣列索引 -> 最近一次有效的 (振幅, 相位)
consecutive_errors = 0  # 連續失敗次數，用於偵測序列埠斷線
last_sample_time = None

# 丟失率追蹤
first_seq = None        # 本次錄製收到的第一個序號（ESP32 不會因開新錄製而重開機，序號是接續的）
last_seq = None         # 上一筆的事件序號
seq_lost = 0            # 序號跳號累計
esp_drop_total = 0      # ESP32 端自報的累計丟棄數（佇列滿 + 長度異常）
csi_lines = 0           # 完整收到且通過長度校驗的 CSI 行數
truncated = 0           # 行被截斷（宣告長度與實得不符）


class TruncatedLine(Exception):
    """CSI 行的宣告長度與實得不符（序列埠壅塞造成的截斷）。

    與一般解析錯誤分開計數：後者要留給真正的非 CSI 雜訊行
    （例如 ESP32 開機時吐出的二進位垃圾），混在一起丟失率的帳就會算錯。
    """

# 韌體帶回的中繼資料統計（用於確認子載波佈局與連線品質）
len_stats = {}        # 原始 CSI 位元組數 -> 出現次數
bb_format_stats = {}  # cur_bb_format -> 出現次數
channel_stats = {}    # (主通道, 次通道) -> 出現次數
rssi_samples = []
meta_reported = False

# 完整 57 條（含已剔除的 DC、+1）的振幅統計，用來核對佈局。
# 正常應只有 DC（索引 28）恆為 0；若換韌體或 ESP-IDF 版本後佈局改變，這裡會先看出來。
raw_amp_sum = np.zeros(EXPECTED_LEN // 2)
raw_zero_count = np.zeros(EXPECTED_LEN // 2, dtype=int)
raw_frames = 0

# ----------------- [建立雙層視覺化畫布] -----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1.5]})
fig.tight_layout(pad=4.0)

ax1.set_title(f"{SUBCARRIERS} Subcarriers True Amplitude "
              f"(ESP32-C5 HT/VHT-LTF 20MHz) — Recording: {ACTION_LABEL}")
ax1.set_xlim(0, WINDOW_SIZE)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Amplitude")
ax1.set_facecolor('#111111')
colors = plt.cm.jet(np.linspace(0, 1, SUBCARRIERS))
lines = []
for i in range(SUBCARRIERS):
    line, = ax1.plot(np.zeros(WINDOW_SIZE), color=colors[i], linewidth=1.5, alpha=0.7)
    lines.append(line)

cax = ax2.imshow(data_matrix, aspect='auto', cmap='jet', vmin=-15, vmax=15)
ax2.set_title("2D CSI Spectrogram (Background Subtracted)")
ax2.set_ylabel(f"Subcarrier ({SUBCARRIER_FREQ[0]} → +{SUBCARRIER_FREQ[-1]})")
ax2.set_xlabel("Time (Frames)")
fig.colorbar(cax, ax=ax2, label="Amplitude Fluctuation")

# ----------------- [資料更新邏輯] -----------------
def update(frame):
    global data_matrix, record_count, error_count, gap_count, last_sample_time
    global meta_reported, layout_skip, consecutive_errors, fwi_count
    global first_seq, last_seq, seq_lost, esp_drop_total, csi_lines, truncated
    global raw_amp_sum, raw_zero_count, raw_frames
    updated = False

    # 序列埠若被拔掉，in_waiting 會直接拋例外，需在迴圈外攔下
    try:
        pending = ser.in_waiting
    except Exception as e:
        print(f"\n[錯誤] 序列埠連線中斷：{e}")
        print("       請檢查 USB 是否被拔除，關閉視窗後重新執行。")
        plt.close(fig)
        return lines + [cax]

    while pending > 0:
        try:
            raw_data = ser.readline().decode('utf-8').strip()
            if raw_data.startswith(("CSI_V3", "CSI_V2", "CSI_DATA")):
                # S3 版韌體的輸出：佈局不同，硬套 C5 的索引會得到錯誤的子載波
                raise ValueError("收到 S3 版韌體格式，請改燒本專案的 C5 firmware.ino")
            if raw_data.startswith(("CSI_V5", "CSI_V4")):
                # CSI_V5,<seq>,<len>,<rssi>,<bb_format>,<dropped>,<channel>,<second>,<fwi>,<esp_us>,<csi...>
                # CSI_V4 同上但沒有 <esp_us>（舊版韌體，CSV 的 EspTime_us 欄留空）
                tokens = raw_data.split(',')
                kind = tokens[0]
                seq = int(tokens[1])
                declared_len = int(tokens[2])
                rssi = int(tokens[3])
                bb_format = int(tokens[4])
                esp_dropped = int(tokens[5])
                channel = int(tokens[6])
                second = int(tokens[7])
                fwi = int(tokens[8])
                if kind == "CSI_V5":
                    esp_us = int(tokens[9])
                    data_start = 10
                else:
                    esp_us = ""
                    data_start = 9
                csi_numbers = [int(x) for x in tokens[data_start:] if x.strip() != ""]

                # 長度與宣告不符代表該行被截斷（序列埠壅塞）。
                # 單獨計數而不併入 error_count——後者要留給真正的非 CSI 雜訊行
                # （例如開機時的二進位垃圾），否則丟失率的帳會算錯。
                if len(csi_numbers) != declared_len:
                    raise TruncatedLine(
                        f"宣告 {declared_len}，實得 {len(csi_numbers)}")
                len_stats[declared_len] = len_stats.get(declared_len, 0) + 1
                bb_format_stats[bb_format] = bb_format_stats.get(bb_format, 0) + 1
                channel_stats[(channel, second)] = channel_stats.get((channel, second), 0) + 1
                rssi_samples.append(rssi)
                fwi_count += fwi

                csi_lines += 1

                # 序號跳號 = 該行在 ESP32 到 PC 之間整行遺失（過去完全偵測不到）
                if last_seq is not None and seq > last_seq + 1:
                    seq_lost += seq - last_seq - 1
                if first_seq is None:
                    first_seq = seq
                last_seq = seq
                esp_drop_total = esp_dropped

                if not meta_reported:
                    meta_reported = True
                    band = "5GHz" if channel > 14 else "2.4GHz"
                    print(f"[韌體] {kind} 格式 | 原始長度 {declared_len} bytes "
                          f"({declared_len // 2} 條子載波) | RSSI {rssi} dBm | "
                          f"{BB_FORMAT_NAMES.get(bb_format, bb_format)} | 通道 {channel} ({band})")
                    print(f"[擷取] HT/VHT-LTF {SUBCARRIERS} 條"
                          f"（子載波 {SUBCARRIER_FREQ[0]}~-1、+1~+{SUBCARRIER_FREQ[-1]}，已剔除 DC）")
                    print(f"[記錄] {'振幅 + 相位' if RECORD_PHASE else '僅振幅'}"
                          f"，CSV 共 {len(COLUMN_NAMES) + 2} 欄")
                    if kind == "CSI_V4":
                        print("[提示] 韌體為 V4，沒有 ESP 端時間戳（EspTime_us 欄留空），"
                              "建議重燒最新 firmware.ino")

                # ESP-IDF 文件：每條子載波兩個 byte，「虛部在前、實部在後」。
                # （S3 版把第一個 byte 當實部，振幅不受影響，但相位等於 π/2 - 真實相位；
                #  分析端對相位做去趨勢後只看起伏，故兩版仍可比，C5 版改為正確順序。）
                amplitudes, phases = [], []
                for i in range(0, len(csi_numbers) - 1, 2):
                    imag, real = csi_numbers[i], csi_numbers[i + 1]
                    amplitudes.append(np.sqrt(real * real + imag * imag))
                    if RECORD_PHASE:
                        phases.append(np.arctan2(imag, real))

                if declared_len != EXPECTED_LEN:
                    # 長度不符代表這幀不是 HT20/VHT20（例如 STBC 228、HT40 234），
                    # 同一陣列索引會對應到不同的實體子載波，故整幀丟棄。
                    layout_skip += 1
                else:
                    raw_amp_sum += amplitudes
                    raw_zero_count += np.asarray(amplitudes) == 0
                    raw_frames += 1

                    # first_word_invalid：前兩條（-28、-27）這幀無效，沿用上一幀的值
                    if fwi:
                        for i in FWI_IDX:
                            if i in last_fwi_vals:
                                amplitudes[i], held_phase = last_fwi_vals[i]
                                if RECORD_PHASE:
                                    phases[i] = held_phase
                    else:
                        for i in FWI_IDX:
                            last_fwi_vals[i] = (amplitudes[i], phases[i] if RECORD_PHASE else None)

                    # 折線圖與熱力圖只顯示振幅，故振幅單獨留一份
                    amp_selected = [amplitudes[i] for i in SUBCARRIER_IDX]
                    row = amp_selected
                    if RECORD_PHASE:
                        row = amp_selected + [phases[i] for i in SUBCARRIER_IDX]

                    # 即時偵測訊號斷點（例如 Wi-Fi/TCP 重連導致的資料空窗）
                    now = time.perf_counter()
                    if last_sample_time is not None:
                        gap = now - last_sample_time
                        if gap > GAP_WARN_THRESHOLD:
                            gap_count += 1
                            print(f"[警告] 偵測到訊號斷點：{gap:.2f}s（發生於第 {record_count + 1} 筆之前）")
                    last_sample_time = now

                    # CSV 寫入（常駐 file handle，定期 flush）
                    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
                    csv_writer.writerow([timestamp, esp_us] + row)
                    record_count += 1
                    if record_count % 50 == 0:
                        csv_file.flush()

                    # 環形緩衝區更新（避免 hstack 重新分配記憶體）
                    data_matrix = np.roll(data_matrix, -1, axis=1)
                    data_matrix[:, -1] = amp_selected
                    updated = True
                    consecutive_errors = 0
        except TruncatedLine:
            # 資料仍在流動，只是這一行壞了，故不計入連續失敗
            truncated += 1
            consecutive_errors = 0
        except Exception as e:
            error_count += 1
            consecutive_errors += 1
            if error_count <= 10:
                print(f"[警告] 第 {error_count} 筆解析錯誤: {e}")
            elif error_count == 11:
                print("[警告] 後續錯誤將不再逐筆顯示，結束時統一報告。")
            # 連續大量失敗且完全沒有成功解析，多半是序列埠斷線或韌體格式不符，
            # 不再無限重試下去（原本會一直刷錯誤訊息且不會結束）
            if consecutive_errors >= 500:
                print(f"\n[錯誤] 連續 {consecutive_errors} 筆解析失敗且無任何成功樣本。")
                print("       可能原因：USB 被拔除、ESP32 重開機、或韌體輸出格式不符。")
                plt.close(fig)
                return lines + [cax]

        try:
            pending = ser.in_waiting
        except Exception as e:
            print(f"\n[錯誤] 序列埠連線中斷：{e}")
            plt.close(fig)
            return lines + [cax]

    if updated:
        # 更新上方折線圖
        for i, line in enumerate(lines):
            line.set_ydata(data_matrix[i, :])

        # 動態 Y 軸（根據最近資料自動伸縮）
        current_max = np.max(data_matrix)
        if current_max > 0:
            ax1.set_ylim(0, current_max * 1.2)

        # 更新下方熱力圖（動態去背）
        baseline = np.mean(data_matrix, axis=1, keepdims=True)
        cax.set_array(data_matrix - baseline)

        # 即時計數器顯示於視窗標題
        fig.canvas.manager.set_window_title(
            f"CSI Monitor - {ACTION_LABEL} | 已錄製 {record_count} 筆 | 丟棄 {error_count} 筆 | 斷點 {gap_count} 次"
        )

    return lines + [cax]

ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    csv_file.flush()
    csv_file.close()
    if 'ser' in locals() and ser.is_open:
        ser.close()
    print(f"\n[結束] 觀測結束。")
    print(f"  - 有效錄製：{record_count} 筆")
    print(f"  - 解析丟棄：{error_count} 筆")
    print(f"  - 訊號斷點：{gap_count} 次（單次間隔 > {GAP_WARN_THRESHOLD}s）")
    if layout_skip:
        print(f"  - 佈局不符丟棄：{layout_skip} 筆（長度不是 {EXPECTED_LEN} bytes，"
              f"多為 STBC 或 40MHz 封包，見下方長度分布）")

    # ---- 丟失率 ----
    # 帳務說明：韌體是先 ++csi_seq 才入佇列，所以「佇列滿而被丟棄」的事件
    # 也已佔用一個序號，會表現為序號跳號。因此序號跳號已涵蓋 ESP32 端的佇列滿丟棄，
    # 兩者不可再相加（相加會重複計算）。
    # 長度異常的丟棄則發生在編號之前，不佔序號，只反映在韌體自報的 dropped 內。
    # 序號要從本次收到的第一筆算起：ESP32 在兩次錄製之間不會重開機，序號是接續的。
    # 舊版用「序號最大值」當總數，實測 3 筆空房間都多算出 18~33 萬筆不存在的「差」。
    if last_seq is not None:
        numbered = last_seq - first_seq + 1   # 本次錄製期間 ESP32 編號的事件數
        print(f"\n  【丟失率分析】")
        print(f"  - ESP32 已編號事件：{numbered} 筆（序號 {first_seq} ~ {last_seq}）")
        print(f"  - 完整收到        ：{csi_lines} 筆")
        print(f"  - 行截斷損壞      ：{truncated} 筆")
        print(f"  - 序號跳號        ：{seq_lost} 筆")
        # 自我核對：三者相加應等於已編號事件數
        accounted = csi_lines + truncated + seq_lost
        mark = "✓ 相符" if accounted == numbered else f"✗ 差 {numbered - accounted}"
        print(f"  - 核對            ：{csi_lines} + {truncated} + {seq_lost} "
              f"= {accounted} vs {numbered}  {mark}")
        print(f"  - 韌體自報丟棄    ：{esp_drop_total} 筆"
              f"（佇列滿已含在序號跳號內；另含長度異常與硬體標記無效，該兩類不佔序號）")
        lost = seq_lost + truncated
        if numbered > 0:
            rate = lost / numbered * 100
            print(f"  - 總丟失率        ：{lost} 筆 ({rate:.2f}%)")
            if lost == 0:
                print("    ✓ 無任何丟失")
            elif rate > 5:
                print("    ⚠ 丟失率偏高。若韌體自報丟棄也大於 0，代表佇列排不及，"
                      "可加大 CSI_QUEUE_LEN 或改二進位輸出降低序列埠負載。")
    else:
        print("\n  【丟失率分析】沒有收到任何 CSI 資料")
    if len_stats:
        total = sum(len_stats.values())
        print(f"  - CSI 原始長度分布：")
        for L, c in sorted(len_stats.items(), key=lambda x: -x[1]):
            print(f"      {L} bytes ({L // 2} 條子載波)：{c} 筆 ({c / total * 100:.1f}%)")
        if len(len_stats) > 1:
            print("      ⚠ 長度不一致代表混入不同型別的封包，同一陣列索引在不同幀")
            print("        會對應到不同的實體子載波，需在韌體端過濾。")
        print(f"  - 封包型別分布：" + "，".join(
            f"{BB_FORMAT_NAMES.get(m, m)} {c} 筆" for m, c in sorted(bb_format_stats.items())))
        print(f"  - 通道分布：" + "，".join(
            f"ch{ch}{'' if sec == 0 else ('+' if sec == 1 else '-')} {c} 筆"
            for (ch, sec), c in sorted(channel_stats.items())))
        if any(sec != 0 for ch, sec in channel_stats):
            print("      ⚠ 出現次通道（40MHz），請把路由器該頻段的頻寬改為 20MHz")
        print(f"  - first_word_invalid：{fwi_count} / {sum(len_stats.values())} 筆"
              f"（這些幀的 -28、-27 沿用上一幀的值）")
    if raw_frames:
        mean_amp = raw_amp_sum / raw_frames
        zero_pct = raw_zero_count / raw_frames * 100
        print(f"  - 佈局核對：完整 {len(mean_amp)} 條的平均振幅（{raw_frames} 筆，依陣列索引）")
        for start in range(0, len(mean_amp), 15):
            seg = range(start, min(start + 15, len(mean_amp)))
            print("      " + " ".join(f"[{i}]{mean_amp[i]:.0f}" for i in seg))
        dead = [i for i in range(len(mean_amp)) if zero_pct[i] > 90]
        print(f"      恆為 0（> 90% 的幀）的索引：{dead or '無'}"
              f"　佈局假設 DC = 索引 {DC_IDX}（只剔除這一條）"
              f"{'　✓ 相符' if dead == [DC_IDX] else '　⚠ 與假設不符，請把這段貼給維護者核對佈局'}")
    if rssi_samples:
        print(f"  - RSSI：平均 {sum(rssi_samples) / len(rssi_samples):.1f} dBm "
              f"(範圍 {min(rssi_samples)} ~ {max(rssi_samples)})")
    print(f"  - 儲存位置：{filename}")
