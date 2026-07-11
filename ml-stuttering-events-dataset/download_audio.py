#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

"""
For each podcast episode:
* Download the raw mp3/m4a file
* Convert it to a 16k mono wav file
# Remove the original file
"""

import os
import pathlib
import subprocess
import socket
import urllib.request
import urllib.error
import urllib.parse

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Download raw audio files for SEP-28k or FluencyBank and convert to 16k hz mono wavs.')
parser.add_argument('--episodes', type=str, required=True,
                   help='Path to the labels csv files (e.g., SEP-28k_episodes.csv)')
parser.add_argument('--wavs', type=str, default="wavs",
                   help='Path where audio files from download_audio.py are saved')


args = parser.parse_args()
episode_uri = args.episodes
wav_dir = args.wavs

# Load episode data
# NOTE: numpy now requires delimiter to be a single character - it used to accept
# ", " directly, but that raises a TypeError on current numpy versions. Split on the
# comma alone, then strip the leftover leading space from every cell.
table = np.loadtxt(episode_uri, dtype=str, delimiter=",")
table = np.char.strip(table)
urls = table[:,2]
n_items = len(urls)

audio_types = [".mp3", ".m4a", ".mp4"]

stats = {
	"already_done": 0,
	"success": 0,
	"download_failed_dns": 0,
	"download_failed_other": 0,
	"recovered_via_curl": 0,
	"ffmpeg_failed": 0,
	"invalid_output": 0,
}


def try_curl_fallback(url, dest_path):
	"""Retry a download via curl.exe when urllib's SSL stack fails on it.

	Some large, modern CDN-backed hosts (WordPress's file CDN, Anchor/Spotify,
	Libsyn, SoundCloud, etc.) serve certificate chains - large Certificate
	Transparency SCT lists, OCSP stapling - that this Python environment's SSL
	stack fails to parse (raising things like "[ASN1: NOT_ENOUGH_DATA]"), even
	though DNS resolution and the TCP connection succeed fine. curl (native to
	Windows 10/11) uses a different, separately-proven-working TLS
	implementation and handles these same URLs correctly - confirmed directly
	earlier against these exact kinds of hosts. Returns True on success."""
	try:
		result = subprocess.run(
			["curl", "-sS", "-L", "-o", str(dest_path), url],
			capture_output=True, text=True, timeout=120
		)
		return result.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
	except (FileNotFoundError, subprocess.TimeoutExpired):
		return False


for i in range(n_items):
	# Get show/episode IDs
	show_abrev = table[i,-2]
	ep_idx = table[i,-1]
	episode_url = table[i,2]

	# Check file extension
	ext = ''
	for ext in audio_types:
		if ext in episode_url:
			break

	# Ensure the base folder exists for this episode
	episode_dir = pathlib.Path(f"{wav_dir}/{show_abrev}/")
	os.makedirs(episode_dir, exist_ok=True)

	# Get file paths
	audio_path_orig = pathlib.Path(f"{episode_dir}/{ep_idx}{ext}")
	wav_path = pathlib.Path(f"{episode_dir}/{ep_idx}.wav")

	# Check if this file has already been downloaded
	if os.path.exists(wav_path):
		stats["already_done"] += 1
		continue

	print("Processing", show_abrev, ep_idx)
	# Download raw audio file. This could be parallelized.
	# Uses urllib instead of shelling out to wget - Windows doesn't ship wget natively
	# (unlike Linux/Mac), so relying on it is fragile depending on what's installed and
	# in PATH. urllib works identically cross-platform and raises a catchable Python
	# exception on failure instead of an opaque shell error we'd have to parse from
	# stderr text.
	download_failed = False
	if not os.path.exists(audio_path_orig):
		try:
			urllib.request.urlretrieve(episode_url, audio_path_orig)
		except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
			# SSL-class errors specifically get one retry via curl before being
			# counted as a real failure - see try_curl_fallback() above for why.
			is_ssl_error = "ssl" in str(e).lower() or "asn1" in str(e).lower() or "certificate" in str(e).lower()
			if is_ssl_error and try_curl_fallback(episode_url, audio_path_orig):
				stats["recovered_via_curl"] += 1
			else:
				# Quick DNS-vs-other classification per failure, so a full run stays
				# informative (worth reviewing afterward) without flooding the output -
				# this is a cheap, compact version of the deeper one-off diagnostic used
				# earlier to actually track down the DNS/ffmpeg/SSL issues.
				parsed_host = urllib.parse.urlparse(episode_url).hostname
				dns_status = "N/A"
				if parsed_host:
					try:
						socket.getaddrinfo(parsed_host, 443)
						dns_status = "DNS OK - failure is NOT resolution (SSL/redirect/HTTP error)"
						stats["download_failed_other"] += 1
					except OSError:
						dns_status = "DNS FAILED - genuine resolution problem for this host"
						stats["download_failed_dns"] += 1
				else:
					stats["download_failed_other"] += 1
				curl_note = " (curl fallback also failed)" if is_ssl_error else ""
				print(f"WARNING: failed to download for {show_abrev} {ep_idx}: {e}  [{dns_status}]{curl_note}")
				print(f"  {episode_url}")
				download_failed = True

	if download_failed:
		# Nothing downloaded, so there's nothing to convert or clean up - move on to
		# the next episode instead of letting a single dead link/SSL quirk/etc. on one
		# host crash or halt the whole batch run.
		continue

	# Convert to 16khz mono wav file.
	# Uses an argument list with subprocess.run instead of a single shell string via
	# shell=True - this sends each argument to the process directly, with no shell
	# involved at all, so there's no possibility of Windows path-separator/quoting
	# ambiguity mangling the command (the same class of problem wget hit). It also
	# lets us capture ffmpeg's actual stdout/stderr, so a failure shows ffmpeg's real
	# error message instead of a generic, unattributed Windows error string.
	ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(audio_path_orig), "-ac", "1", "-ar", "16000", str(wav_path)]
	try:
		result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
		if result.returncode != 0:
			print(f"WARNING: ffmpeg failed (exit code {result.returncode}) on {audio_path_orig}")
			print(f"  Command: {' '.join(ffmpeg_cmd)}")
			print(f"  ffmpeg stderr (last 500 chars): ...{result.stderr[-500:]}")
			stats["ffmpeg_failed"] += 1
	except FileNotFoundError:
		print("FATAL: 'ffmpeg' was not found - it isn't installed, or isn't on PATH.")
		print("  Verify with: ffmpeg -version")
		print("  If that also fails to run, install ffmpeg and make sure its folder")
		print("  is added to your PATH environment variable, then restart the terminal.")
		raise

	# Verify the conversion actually produced usable audio before treating this as
	# done - without this check, a failed download (e.g. a dead link returning an
	# HTML error page instead of audio) can silently produce an empty/near-empty wav
	# that the "if os.path.exists(wav_path): continue" check above would treat as
	# already-downloaded forever, never retrying it. This is a likely explanation for
	# the 0:00-duration files seen in the Kaggle copy of this dataset.
	MIN_VALID_DURATION_SEC = 1.0
	valid = False
	if os.path.exists(wav_path):
		try:
			probe = subprocess.run(
				["ffprobe", "-v", "error", "-show_entries", "format=duration",
				 "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
				capture_output=True, text=True
			)
			duration = float(probe.stdout.strip())
			valid = duration >= MIN_VALID_DURATION_SEC
		except (ValueError, FileNotFoundError):
			valid = False

	if not valid:
		print(f"WARNING: {wav_path} appears invalid (missing or too short) - "
			  f"removing so it gets retried on the next run instead of being "
			  f"silently skipped as already-downloaded.")
		if os.path.exists(wav_path):
			os.remove(wav_path)
		stats["invalid_output"] += 1
	else:
		stats["success"] += 1

	# Remove the original mp3/m4a file, if it exists - guarded, since a failed
	# ffmpeg conversion (as opposed to a failed download, already handled above)
	# shouldn't crash cleanup for the rest of the batch either.
	if os.path.exists(audio_path_orig):
		os.remove(audio_path_orig)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Already done (skipped):     {stats['already_done']}")
print(f"Newly converted (success):  {stats['success']}")
print(f"Recovered via curl (SSL):   {stats['recovered_via_curl']}")
print(f"Download failed - DNS:      {stats['download_failed_dns']}")
print(f"Download failed - other:    {stats['download_failed_other']}  (SSL/HTTP/redirect - see warnings above)")
print(f"ffmpeg conversion failed:   {stats['ffmpeg_failed']}")
print(f"Invalid output (too short): {stats['invalid_output']}")
print("=" * 60)