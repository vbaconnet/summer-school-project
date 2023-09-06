echo "Running cuda"
echo "1/8..."
./energy_storms_cuda 35 test_files/test_01_a35_p8_w1 test_files/test_01_a35_p7_w2 test_files/test_01_a35_p5_w3 test_files/test_01_a35_p8_w4 > times_cuda.txt
echo "2/8..."
./energy_storms_cuda 30000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 >> times_cuda.txt

echo "3/8..."
./energy_storms_cuda 20 test_files/test_03_a20_p4_w1 >> times_cuda.txt

echo "4/8..."
./energy_storms_cuda 20 test_files/test_04_a20_p4_w1 >> times_cuda.txt

echo "5/8..."
./energy_storms_cuda 20 test_files/test_05_a20_p4_w1 >> times_cuda.txt

echo "6/8..."
./energy_storms_cuda 20 test_files/test_06_a20_p4_w1 >> times_cuda.txt

echo "7/8..."
./energy_storms_cuda 1000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4 >> times_cuda.txt

echo "8/8..."
./energy_storms_cuda 100000000 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3 >> times_cuda.txt

