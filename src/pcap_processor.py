# ============================================================
# PCAP PROCESSOR — Extracts features from PCAP files
# and feeds them to our XGBoost model via Flask API
#
# v2: Uses tshark for bulk packet extraction (50x faster than pyshark)
#     Handles 50k-packet nmap scans in seconds instead of minutes.
#
# Flow splitting rules:
#   1. New SYN packet (SYN set, ACK not set) → new sub-flow
#   2. Time gap > FLOW_TIMEOUT seconds → new sub-flow
# ============================================================

import subprocess
import numpy as np
import pandas as pd
import requests
import joblib
import os
import time
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\data"
MODELS_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\models"
PCAP_FOLDER = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\captures"

FLASK_URL    = "http://localhost:5000/api/live"
FLOW_TIMEOUT = 2.0   # seconds — gap larger than this = new flow

# ── Load artifacts ─────────────────────────────────────────────
print("Loading model artifacts...")
le           = joblib.load(os.path.join(DATA_PATH,   'label_encoder.pkl'))
scaler       = joblib.load(os.path.join(DATA_PATH,   'scaler.pkl'))
feature_cols = joblib.load(os.path.join(DATA_PATH,   'feature_columns.pkl'))
print(f"✅ Artifacts loaded  —  expecting {len(feature_cols)} features")

os.makedirs(PCAP_FOLDER, exist_ok=True)


# ============================================================
# ACTIVE / IDLE INTERVAL COMPUTATION
# ============================================================
ACTIVITY_TIMEOUT = 1.0

def compute_active_idle(timestamps):
    if len(timestamps) < 2:
        zero = {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
        return zero, zero

    active_durations, idle_durations = [], []
    burst_start = timestamps[0]
    prev_ts     = timestamps[0]

    for ts in timestamps[1:]:
        gap = ts - prev_ts
        if gap >= ACTIVITY_TIMEOUT:
            active_durations.append((prev_ts - burst_start) * 1e6)
            idle_durations.append(gap * 1e6)
            burst_start = ts
        prev_ts = ts
    active_durations.append((prev_ts - burst_start) * 1e6)

    def stats(lst):
        if not lst:
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
        return {'mean': float(np.mean(lst)),  'std': float(np.std(lst)),
                'max':  float(np.max(lst)),   'min': float(np.min(lst))}

    return stats(active_durations), stats(idle_durations)


def _get_init_win(pkt_list):
    """Return TCP window size from first packet, or 0 if unavailable."""
    if not pkt_list:
        return 0
    return pkt_list[0].get('window', 0)


# ============================================================
# TSHARK BULK PACKET EXTRACTION
# ============================================================
def read_packets_tshark(pcap_file):
    """
    Use tshark to dump all packets as CSV in one subprocess call.
    Returns list of packet dicts. ~50x faster than pyshark for large files.
    """
    fields = [
        'frame.time_epoch',   # timestamp
        'ip.src',             # source IP
        'ip.dst',             # dest IP
        'tcp.srcport',        # TCP src port
        'tcp.dstport',        # TCP dst port
        'udp.srcport',        # UDP src port
        'udp.dstport',        # UDP dst port
        'ip.proto',           # protocol (6=TCP, 17=UDP)
        'ip.len',             # IP payload length (matches CICIDS2017)
        'tcp.flags',          # TCP flags as hex
        'tcp.window_size_value',  # TCP window size
    ]

    cmd = [
        'tshark',
        '-r', pcap_file,
        '-T', 'fields',
        '-E', 'separator=|',
        '-E', 'occurrence=f',   # first occurrence of each field
    ] + [arg for f in fields for arg in ('-e', f)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError:
        print("❌ tshark not found! Install Wireshark/tshark and add to PATH.")
        print("   Windows: add C:\\Program Files\\Wireshark to PATH")
        return []
    except subprocess.TimeoutExpired:
        print("❌ tshark timed out after 120s")
        return []

    if result.returncode != 0:
        print(f"❌ tshark error: {result.stderr[:200]}")
        return []

    packets = []
    for line in result.stdout.splitlines():
        parts = line.split('|')
        if len(parts) < 4:
            continue
        try:
            ts       = float(parts[0]) if parts[0] else None
            src_ip   = parts[1]
            dst_ip   = parts[2]
            tcp_sp   = parts[3]
            tcp_dp   = parts[4]
            udp_sp   = parts[5]
            udp_dp   = parts[6]
            proto    = parts[7]
            ip_len   = parts[8]
            tcp_flg  = parts[9]
            tcp_win  = parts[10] if len(parts) > 10 else ''

            if not ts or not src_ip or not dst_ip:
                continue

            # Determine protocol and ports
            if proto == '6' and tcp_sp and tcp_dp:
                src_port = int(tcp_sp)
                dst_port = int(tcp_dp)
                transport = 'TCP'
            elif proto == '17' and udp_sp and udp_dp:
                src_port = int(udp_sp)
                dst_port = int(udp_dp)
                transport = 'UDP'
            else:
                continue

            # Parse TCP flags
            flags = 0
            if tcp_flg:
                try:
                    flags = int(tcp_flg, 16)
                except (ValueError, TypeError):
                    flags = 0

            # Parse window size
            window = 0
            if tcp_win:
                try:
                    window = int(tcp_win)
                except (ValueError, TypeError):
                    window = 0

            # Parse IP length
            length = 0
            if ip_len:
                try:
                    length = int(ip_len)
                except (ValueError, TypeError):
                    length = 0

            packets.append({
                'ts':        ts,
                'src_ip':    src_ip,
                'dst_ip':    dst_ip,
                'src_port':  src_port,
                'dst_port':  dst_port,
                'transport': transport,
                'length':    length,
                'flags':     flags,
                'window':    window,
            })

        except (ValueError, IndexError):
            continue

    return packets


# ============================================================
# FLOW RECONSTRUCTION WITH SYN-BASED SPLITTING
# ============================================================
def build_flows(packets):
    """Group packets into bidirectional flows with SYN-based splitting."""
    flows         = {}
    conn_counter  = defaultdict(int)
    last_pkt_time = {}
    flow_init     = {}

    for pkt in packets:
        src_ip    = pkt['src_ip']
        dst_ip    = pkt['dst_ip']
        src_port  = pkt['src_port']
        dst_port  = pkt['dst_port']
        proto     = pkt['transport']
        ts        = pkt['ts']
        flags     = pkt['flags']

        is_syn = bool(flags & 0x02) and not bool(flags & 0x10)

        fwd_base = (src_ip, dst_ip, src_port, dst_port, proto)
        rev_base = (dst_ip, src_ip, dst_port, src_port, proto)

        if fwd_base in last_pkt_time:
            base_key  = fwd_base
            init_ip   = src_ip
            init_port = src_port
        elif rev_base in last_pkt_time:
            base_key  = rev_base
            init_ip   = rev_base[0]
            init_port = rev_base[2]
        else:
            base_key  = fwd_base
            init_ip   = src_ip
            init_port = src_port

        prev_ts = last_pkt_time.get(base_key, ts)
        gap     = ts - prev_ts

        if is_syn or gap > FLOW_TIMEOUT:
            if base_key in last_pkt_time:
                conn_counter[base_key] += 1

        last_pkt_time[base_key] = ts

        conn_id  = conn_counter[base_key]
        full_key = base_key + (conn_id,)

        if full_key not in flow_init:
            flow_init[full_key] = (init_ip, init_port)

        real_init_ip, real_init_port = flow_init[full_key]
        direction = ('fwd'
                     if (src_ip == real_init_ip and src_port == real_init_port)
                     else 'bwd')

        if full_key not in flows:
            flows[full_key] = []
        flows[full_key].append({
            'ts':        ts,
            'length':    pkt['length'],
            'flags':     flags,
            'direction': direction,
            'window':    pkt['window'],
        })

    return flows


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def compute_features(flows):
    """Convert flow packet lists into CICIDS2017 feature dicts."""
    flow_features = []

    for full_key, packets in flows.items():
        if len(packets) < 1:
            continue

        # Server port = lower of the two ports (heuristic)
        dst_port_val = min(full_key[2], full_key[3])

        packets.sort(key=lambda x: x['ts'])
        timestamps = [p['ts']     for p in packets]
        lengths    = [p['length'] for p in packets]
        flags_list = [p['flags']  for p in packets]

        fwd_pkts = [p for p in packets if p['direction'] == 'fwd']
        bwd_pkts = [p for p in packets if p['direction'] == 'bwd']

        fwd_lengths = [p['length'] for p in fwd_pkts] or [0]
        bwd_lengths = [p['length'] for p in bwd_pkts] or [0]

        def iats_us(ts_list):
            if len(ts_list) < 2:
                return [0]
            return [(ts_list[i+1] - ts_list[i]) * 1e6
                    for i in range(len(ts_list) - 1)]

        all_iats  = iats_us(timestamps)
        fwd_times = [p['ts'] for p in fwd_pkts]
        bwd_times = [p['ts'] for p in bwd_pkts]
        fwd_iats  = iats_us(fwd_times)
        bwd_iats  = iats_us(bwd_times)

        flow_duration   = max((timestamps[-1] - timestamps[0]) * 1e6, 1.0)
        total_fwd_bytes = sum(fwd_lengths)
        total_bwd_bytes = sum(bwd_lengths)
        total_bytes     = total_fwd_bytes + total_bwd_bytes

        fin_count = sum(1 for f in flags_list if f & 0x01)
        syn_count = sum(1 for f in flags_list if f & 0x02)
        rst_count = sum(1 for f in flags_list if f & 0x04)
        psh_count = sum(1 for f in flags_list if f & 0x08)
        ack_count = sum(1 for f in flags_list if f & 0x10)
        urg_count = sum(1 for f in flags_list if f & 0x20)
        cwe_count = sum(1 for f in flags_list if f & 0x40)
        ece_count = sum(1 for f in flags_list if f & 0x80)

        active_stats, idle_stats = compute_active_idle(timestamps)

        features = {
            'Destination Port':              dst_port_val,
            'Flow Duration':                 flow_duration,
            'Total Fwd Packets':             len(fwd_pkts),
            'Total Backward Packets':        len(bwd_pkts),
            'Total Length of Fwd Packets':   total_fwd_bytes,
            'Total Length of Bwd Packets':   total_bwd_bytes,
            'Fwd Packet Length Max':         max(fwd_lengths),
            'Fwd Packet Length Min':         min(fwd_lengths),
            'Fwd Packet Length Mean':        float(np.mean(fwd_lengths)),
            'Fwd Packet Length Std':         float(np.std(fwd_lengths)),
            'Bwd Packet Length Max':         max(bwd_lengths),
            'Bwd Packet Length Min':         min(bwd_lengths),
            'Bwd Packet Length Mean':        float(np.mean(bwd_lengths)),
            'Bwd Packet Length Std':         float(np.std(bwd_lengths)),
            'Flow Bytes/s':                  total_bytes     / flow_duration * 1e6,
            'Flow Packets/s':                len(packets)    / flow_duration * 1e6,
            'Flow IAT Mean':                 float(np.mean(all_iats)),
            'Flow IAT Std':                  float(np.std(all_iats)),
            'Flow IAT Max':                  float(np.max(all_iats)),
            'Flow IAT Min':                  float(np.min(all_iats)),
            'Fwd IAT Total':                 float(sum(fwd_iats)),
            'Fwd IAT Mean':                  float(np.mean(fwd_iats)),
            'Fwd IAT Std':                   float(np.std(fwd_iats)),
            'Fwd IAT Max':                   float(np.max(fwd_iats)),
            'Fwd IAT Min':                   float(np.min(fwd_iats)),
            'Bwd IAT Total':                 float(sum(bwd_iats)),
            'Bwd IAT Mean':                  float(np.mean(bwd_iats)),
            'Bwd IAT Std':                   float(np.std(bwd_iats)),
            'Bwd IAT Max':                   float(np.max(bwd_iats)),
            'Bwd IAT Min':                   float(np.min(bwd_iats)),
            'Fwd PSH Flags':                 psh_count,
            'Fwd URG Flags':                 urg_count,
            'Fwd Header Length':             len(fwd_pkts) * 20,
            'Bwd Header Length':             len(bwd_pkts) * 20,
            'Fwd Packets/s':                 len(fwd_pkts)  / flow_duration * 1e6,
            'Bwd Packets/s':                 len(bwd_pkts)  / flow_duration * 1e6,
            'Min Packet Length':             min(lengths),
            'Max Packet Length':             max(lengths),
            'Packet Length Mean':            float(np.mean(lengths)),
            'Packet Length Std':             float(np.std(lengths)),
            'Packet Length Variance':        float(np.var(lengths)),
            'FIN Flag Count':                fin_count,
            'SYN Flag Count':                syn_count,
            'RST Flag Count':                rst_count,
            'PSH Flag Count':                psh_count,
            'ACK Flag Count':                ack_count,
            'URG Flag Count':                urg_count,
            'CWE Flag Count':                cwe_count,
            'ECE Flag Count':                ece_count,
            'Down/Up Ratio':                 len(bwd_pkts) / max(len(fwd_pkts), 1),
            'Average Packet Size':           float(np.mean(lengths)),
            'Avg Fwd Segment Size':          float(np.mean(fwd_lengths)),
            'Avg Bwd Segment Size':          float(np.mean(bwd_lengths)),
            'Fwd Header Length.1':           len(fwd_pkts) * 20,
            'Subflow Fwd Packets':           len(fwd_pkts),
            'Subflow Fwd Bytes':             total_fwd_bytes,
            'Subflow Bwd Packets':           len(bwd_pkts),
            'Subflow BwdBytes':             total_bwd_bytes,
            'Init_Win_bytes_forward':        _get_init_win(fwd_pkts),
            'Init_Win_bytes_backward':       _get_init_win(bwd_pkts),
            'act_data_pkt_fwd':              len([p for p in fwd_pkts if p['length'] > 0]),
            'min_seg_size_forward':          min(fwd_lengths),
            'Active Mean':                   active_stats['mean'],
            'Active Std':                    active_stats['std'],
            'Active Max':                    active_stats['max'],
            'Active Min':                    active_stats['min'],
            'Idle Mean':                     idle_stats['mean'],
            'Idle Std':                      idle_stats['std'],
            'Idle Max':                      idle_stats['max'],
            'Idle Min':                      idle_stats['min'],
        }

        flow_features.append(features)

    return flow_features


def extract_flows(pcap_file):
    """Main entry point: pcap → feature dicts."""
    print(f"\n📂 Processing: {pcap_file}")
    t0 = time.time()

    packets = read_packets_tshark(pcap_file)
    if not packets:
        return []

    print(f"   tshark parsed {len(packets)} packets in {time.time()-t0:.1f}s")

    flows = build_flows(packets)
    print(f"   → {len(flows)} flows reconstructed (SYN-split)")

    features = compute_features(flows)
    print(f"   → {len(features)} flows ready for prediction")

    # Debug: show first flow's key values
    if features:
        f0 = features[0]
        print(f"   [DEBUG] First flow: DstPort={f0['Destination Port']}  "
              f"Duration={f0['Flow Duration']:.0f}µs  "
              f"Fwd={f0['Total Fwd Packets']}pkts  "
              f"Bwd={f0['Total Backward Packets']}pkts  "
              f"SYN={f0['SYN Flag Count']}  RST={f0['RST Flag Count']}")

    return features


# ============================================================
# PREDICT AND SEND TO FLASK
# ============================================================
def predict_flows(flow_features):
    if not flow_features:
        print("⚠️  No flows to predict")
        return

    print(f"\n🔍 Running predictions on {len(flow_features)} flows...")

    results    = []
    skip_count = 0

    for i, features in enumerate(flow_features):
        feature_vector = [float(features.get(col, 0)) for col in feature_cols]
        feature_array  = np.array([feature_vector])
        feature_array  = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        if feature_array.shape[1] != len(feature_cols):
            skip_count += 1
            continue

        feature_df     = pd.DataFrame(feature_array, columns=feature_cols)
        feature_scaled = scaler.transform(feature_df)

        payload = {'features': feature_scaled[0].tolist(), 'flow_index': i + 1}

        try:
            resp   = requests.post(FLASK_URL, json=payload, timeout=5)
            result = resp.json()
            results.append(result)

            label = result.get('prediction', 'Unknown')
            conf  = result.get('confidence', 0)

            if label != 'BENIGN':
                print(f"  🚨 Flow {i+1:4d}: {label:35s} ({conf:.1f}%)")
            else:
                print(f"  ✅ Flow {i+1:4d}: BENIGN ({conf:.1f}%)")

        except Exception as e:
            print(f"  ❌ Error on flow {i+1}: {e}")

        time.sleep(0.02)

    attacks = sum(1 for r in results if r.get('is_attack'))
    total   = len(results)

    print(f"\n{'='*52}")
    print(f"  DETECTION SUMMARY")
    print(f"{'='*52}")
    print(f"  Total flows    : {total}")
    print(f"  Attacks found  : {attacks}")
    print(f"  Normal traffic : {total - attacks}")
    print(f"  Detection rate : {attacks / max(total, 1) * 100:.1f}%")
    if skip_count:
        print(f"  Skipped (bad)  : {skip_count}")
    print(f"{'='*52}\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 52)
    print("  LIVE ATTACK DETECTION SYSTEM  (tshark edition)")
    print("=" * 52)
    print(f"\n📁 Drop PCAP files into:\n   {PCAP_FOLDER}")
    print(f"\n🌐 Sending results to:\n   {FLASK_URL}")
    print(f"\n💡 Recommended Kali attacks:")
    print(f"   PortScan : nmap -Pn -sS -p 1-65535 --min-rate 1000 <laptop-ip>")
    print(f"   SSH BF   : hydra -l root -P /usr/share/wordlists/rockyou.txt ssh://<laptop-ip> -t 4 -v")
    print(f"\nPress Ctrl+C to stop\n")

    processed_files = set()

    while True:
        try:
            current_files = set(
                f for f in os.listdir(PCAP_FOLDER)
                if f.endswith(('.pcap', '.pcapng', '.cap'))
            )

            for filename in sorted(current_files - processed_files):
                filepath = os.path.join(PCAP_FOLDER, filename)
                print(f"\n📡 New file detected: {filename}")
                time.sleep(2)
                flows = extract_flows(filepath)
                predict_flows(flows)
                processed_files.add(filename)

            time.sleep(2)

        except KeyboardInterrupt:
            print("\n👋 Stopped.")
            break