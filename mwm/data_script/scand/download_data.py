import os
import time
from pyDataverse.api import NativeApi, DataAccessApi
from pyDataverse.models import Dataverse
import os, requests, sys


class LowSpeedError(Exception):
    pass


def flush_print(content: str) -> None:
    sys.stdout.write(content)
    sys.stdout.flush()


def split_into_n_lists(files_list, n=5):
    L = len(files_list)
    q, r = divmod(L, n)  # q: 每份至少 q 个，前 r 份多 1 个
    parts = []
    start = 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        parts.append(files_list[start:start + size])
        start += size
    return parts


def download_file_show_speed(download_url, headers, dir, filename):
    max_retries = 100
    max_low_speed_time = 30
    backoff_factor = 1
    min_MBs = 1

    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[SKIP] exists: {out_path}", flush=True)
        return

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            last_t = t0
            last_bytes = 0
            downloaded = 0
            low_speed_time = 0

            with requests.get(download_url, headers=headers, stream=True, timeout=180) as response:
                response.raise_for_status()

                total = response.headers.get("Content-Length")
                total = int(total) if total and total.isdigit() else None

                with open(out_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_t >= 1.0:
                            inst_speed = (downloaded - last_bytes) / (now - last_t)  # B/s
                            avg_speed = downloaded / (now - t0) if now > t0 else 0.0

                            if inst_speed/1024/1024 < min_MBs:
                                low_speed_time += 1
                            else:
                                low_speed_time = 0

                            if low_speed_time > max_low_speed_time:
                                raise LowSpeedError(f"Download too slow: {inst_speed/1024/1024:.3f} MB/s < {min_MBs} MB/s for {low_speed_time}s")

                            if total:
                                pct = downloaded * 100.0 / total
                                flush_print(
                                    f"\r[{filename}] {pct:6.2f}%  "
                                    f"{downloaded/1024/1024:8.2f}MB/{total/1024/1024:8.2f}MB  "
                                    f"inst {inst_speed/1024/1024:6.2f}MB/s  avg {avg_speed/1024/1024:6.2f}MB/s"
                                )
                            else:
                                flush_print(
                                    f"\r[{filename}] {downloaded/1024/1024:8.2f}MB  "
                                    f"inst {inst_speed/1024/1024:6.2f}MB/s  avg {avg_speed/1024/1024:6.2f}MB/s"
                                )

                            last_t = now
                            last_bytes = downloaded

            dt = time.time() - t0
            avg_speed = downloaded / dt if dt > 0 else 0.0
            print(f"\nSuccessfully downloaded: {filename}  ({downloaded/1024/1024:.2f}MB, avg {avg_speed/1024/1024:.2f}MB/s)")
            return

        except Exception as e:
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                print("remove error!!!!!!!!!!!!!!!!!!!!!!!!!!")
                pass

            wait_time = 10
            print(f"Error downloading {filename}: {str(e)}. Retrying in {wait_time} seconds...", flush=True)
            time.sleep(wait_time)

    print(f"Failed to download {filename} after {max_retries} attempts.", flush=True)


# Function to download a file with retry mechanism and exponential backoff
def download_file(download_url, headers, dir, filename):
    max_retries = 5
    backoff_factor = 1
    for attempt in range(max_retries):
        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=180) as response:
                response.raise_for_status()
                with open(os.path.join(dir, filename), "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded: {filename}")
            return
        except requests.RequestException as e:
            wait_time = backoff_factor * (2 ** attempt)
            print(f"Error downloading {filename}: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    print(f"Failed to download {filename} after {max_retries} attempts.")


def main(part_idx, parts, outdir):
    
    for file in parts[part_idx]:
        filename = file["dataFile"]["filename"]
        file_id = file["dataFile"]["id"]
        
        if "DELIVERY" in filename or (not filename.lower().endswith(".bag")):
            continue

        print(f"Downloading: File name {filename}, id {file_id}")

        # Construct the download URL
        download_url = f"{base_url}api/access/datafile/{file_id}"

        # Set up the request headers with the API token\
        headers = {}
        token = os.environ.get("DATAVERSE_KEY", "").strip()
        if token:
            headers["X-Dataverse-key"] = token

        # Download the file
        download_file_show_speed(download_url, headers, outdir, filename)

    print("Download process completed.")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, help="which split to download (0-based)")
    args = parser.parse_args()

    base_url = 'https://dataverse.tdl.org/'
    api_token = "5ddf7cd3-012d-4892-ad4b-717a2aff92ca"
    api = NativeApi(base_url, api_token)
    data_api = DataAccessApi(base_url)

    DOI = "doi:10.18738/T8/0PRYRH"
    dataset = api.get_dataset(DOI)
    
    outdir = f"random_mdps{args.part}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists("delivery_mdp"):
        os.makedirs("delivery_mdp")

    files_list = dataset.json()['data']['latestVersion']['files']
    parts = split_into_n_lists(files_list, n=5)

    main(args.part, parts, outdir)


