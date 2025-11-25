"""
Convert datamart_full.xlsx to CSV for 10-20x faster loading

Excel files are very slow to read. This script converts to CSV which is much faster.
Run this once, then use data.csv for training.
"""
import pandas as pd
import time

print("="*60)
print("Converting datamart_full.xlsx to CSV format...")
print("="*60)

# Read Excel (this will be slow - only need to do once!)
print("\n[1/3] Reading Excel file... (this may take 20-30 seconds)")
start_time = time.time()
df = pd.read_excel('Data/datamart_full.xlsx')
excel_time = time.time() - start_time
print(f"[OK] Excel loaded in {excel_time:.1f} seconds")
print(f"     Data shape: {df.shape}")

# Save to CSV
print("\n[2/3] Saving to CSV...")
start_time = time.time()
df.to_csv('Data/data.csv', index=False)
csv_save_time = time.time() - start_time
print(f"[OK] CSV saved in {csv_save_time:.1f} seconds")

# # Test CSV reading speed
# print("\n[3/3] Testing CSV read speed...")
# start_time = time.time()
# df_test = pd.read_csv('Data/data.csv')
# csv_read_time = time.time() - start_time
# print(f"✓ CSV loaded in {csv_read_time:.1f} seconds")

# # Speed comparison
# print("\n" + "="*60)
# print("SPEED COMPARISON:")
# print("="*60)
# print(f"Excel reading:  {excel_time:.1f} seconds")
# print(f"CSV reading:    {csv_read_time:.1f} seconds")
# print(f"Speedup:        {excel_time/csv_read_time:.1f}x faster!")
# print("\n✓ Conversion complete! Use 'Data/data.csv' for training.")
# print("="*60)