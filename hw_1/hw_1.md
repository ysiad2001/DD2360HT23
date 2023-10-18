# Assignment I: GPU programming enviroment

## Observation

1. When the transfer size is adequately large, the device-to-device bandwidth is significantly larger than host-to-device and device-to-host bandwidth. This is reasonable since device-to-device data transfer can be done in parallel, while other data transfer cannot.
2. When transfer size reaches higher value, the bandwidth tends to converge to a limit. This is also reasonable since when the data is too much, the data bus might be more occupied, thus increasing the transfer size does not increase the bandwidth significantly.

## Bandwidth test

```
!./bandwidthTest/bandwidthTest
```

```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla T4
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   32000000			12.4

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   32000000			13.1

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   32000000			239.4

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

## Shmoo Mode bandwidth test

```
!./bandwidthTest/bandwidthTest --mode=shmoo
```
```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla T4
 Shmoo Mode

.................................................................................
 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   1000				0.4
   2000				0.7
   3000				1.1
   4000				1.5
   5000				1.8
   6000				2.0
   7000				2.3
   8000				2.5
   9000				2.7
   10000			2.9
   11000			3.1
   12000			3.3
   13000			3.6
   14000			3.7
   15000			3.8
   16000			4.0
   17000			4.2
   18000			4.2
   19000			4.2
   20000			4.5
   22000			4.7
   24000			4.9
   26000			5.1
   28000			5.3
   30000			5.5
   32000			5.8
   34000			5.8
   36000			6.2
   38000			6.4
   40000			6.4
   42000			6.6
   44000			6.7
   46000			6.9
   48000			7.0
   50000			7.1
   60000			7.6
   70000			8.0
   80000			8.5
   90000			8.8
   100000			9.1
   200000			10.4
   300000			11.1
   400000			11.4
   500000			11.5
   600000			11.7
   700000			9.0
   800000			11.6
   900000			9.5
   1000000			11.8
   2000000			10.7
   3000000			11.1
   4000000			11.5
   5000000			11.6
   6000000			11.8
   7000000			11.9
   8000000			12.0
   9000000			11.9
   10000000			12.0
   11000000			12.0
   12000000			12.1
   13000000			12.1
   14000000			12.1
   15000000			12.1
   16000000			12.1
   18000000			12.2
   20000000			12.2
   22000000			12.2
   24000000			12.2
   26000000			12.2
   28000000			12.2
   30000000			12.2
   32000000			12.3
   36000000			12.3
   40000000			12.3
   44000000			12.3
   48000000			12.3
   52000000			12.3
   56000000			12.3
   60000000			12.3
   64000000			12.3
   68000000			12.3

.................................................................................
 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   1000				0.6
   2000				1.2
   3000				1.8
   4000				1.4
   5000				2.8
   6000				3.2
   7000				3.9
   8000				4.0
   9000				4.7
   10000			5.1
   11000			5.2
   12000			5.6
   13000			5.8
   14000			6.1
   15000			6.3
   16000			6.5
   17000			6.7
   18000			6.8
   19000			7.1
   20000			7.3
   22000			7.6
   24000			7.9
   26000			8.1
   28000			8.4
   30000			8.6
   32000			8.8
   34000			8.7
   36000			9.1
   38000			9.2
   40000			9.4
   42000			9.5
   44000			9.6
   46000			9.7
   48000			9.9
   50000			9.8
   60000			10.3
   70000			10.7
   80000			10.7
   90000			11.2
   100000			11.3
   200000			10.8
   300000			9.5
   400000			11.0
   500000			11.5
   600000			12.8
   700000			12.5
   800000			12.0
   900000			11.6
   1000000			11.1
   2000000			12.9
   3000000			13.0
   4000000			13.0
   5000000			13.1
   6000000			13.0
   7000000			13.1
   8000000			13.1
   9000000			13.1
   10000000			13.1
   11000000			13.1
   12000000			13.1
   13000000			13.1
   14000000			13.1
   15000000			13.1
   16000000			13.1
   18000000			13.1
   20000000			13.1
   22000000			13.1
   24000000			13.1
   26000000			13.1
   28000000			13.1
   30000000			13.1
   32000000			13.2
   36000000			13.2
   40000000			13.2
   44000000			13.2
   48000000			13.2
   52000000			13.2
   56000000			13.2
   60000000			13.2
   64000000			13.2
   68000000			13.2

.................................................................................
 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   1000				0.6
   2000				1.2
   3000				1.6
   4000				2.3
   5000				1.9
   6000				3.5
   7000				4.1
   8000				4.7
   9000				5.4
   10000			5.7
   11000			6.6
   12000			6.7
   13000			7.7
   14000			8.0
   15000			8.8
   16000			9.5
   17000			9.5
   18000			10.6
   19000			11.0
   20000			11.6
   22000			12.3
   24000			13.9
   26000			15.8
   28000			15.8
   30000			18.0
   32000			18.1
   34000			17.1
   36000			14.2
   38000			17.8
   40000			22.8
   42000			25.6
   44000			26.9
   46000			26.3
   48000			26.9
   50000			30.3
   60000			35.3
   70000			42.7
   80000			46.7
   90000			54.0
   100000			61.2
   200000			124.2
   300000			186.9
   400000			265.4
   500000			341.0
   600000			413.5
   700000			271.0
   800000			468.6
   900000			490.4
   1000000			516.9
   2000000			580.1
   3000000			207.8
   4000000			214.7
   5000000			220.9
   6000000			223.1
   7000000			226.5
   8000000			228.4
   9000000			229.8
   10000000			231.1
   11000000			232.2
   12000000			232.8
   13000000			234.1
   14000000			234.7
   15000000			235.3
   16000000			235.5
   18000000			236.4
   20000000			237.2
   22000000			237.9
   24000000			238.0
   26000000			238.2
   28000000			238.5
   30000000			239.2
   32000000			239.4
   36000000			239.8
   40000000			240.1
   44000000			240.4
   48000000			240.5
   52000000			240.9
   56000000			241.0
   60000000			241.1
   64000000			242.2
   68000000			241.4

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

