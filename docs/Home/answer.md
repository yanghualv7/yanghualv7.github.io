```c++
//eg1能进入if，eg2不能进入if,暂时还不清楚为什么

typedef unsigned char  UINT8_T;
typedef unsigned short UINT16_T;
typedef unsigned long  UINT32_T;

UINT16_T i = 0
UINT8_T raw_rsp_data[MAX_RESP_LEN] = { 0 };
UINT32_T rsp_data[]

eg1:
int num = raw_rsp_data[3] + 5;
int data_num_0 = 0;
int data_num_1 = 0;	
while (i < num)
	{
		data_num_0 = raw_rsp_data[i];
		if (data_num_0 == 0xf3)
		{
			data_num_1 = raw_cmd_data[i + 1];
			if (data_num_1 > 5)
			{
				rsp_data[j] = raw_rsp_data[i];
				j++;
			}
			else
				num++;
		}
		else
		{
			rsp_data[j] = raw_rsp_data[i];
			j++;
		}
		i++;
	}

eg2:
int num = raw_rsp_data[3] + 5;
	while (i < num)
	{
		if (raw_rsp_data[i] == 0xf3)
		{
			data_num_1 = raw_cmd_data[i + 1];
			if (raw_rsp_data[i+1] > 5)
			{
				rsp_data[j] = raw_rsp_data[i];
				j++;
			}
			else
				num++;
		}
		else
		{
			rsp_data[j] = raw_rsp_data[i];
			j++;
		}  
		i++;
	}
```