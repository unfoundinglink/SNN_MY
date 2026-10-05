import tkinter as tk
from tkinter import filedialog
import os
import csv
import xlrd
import xlwt
import random
import matplotlib.pyplot as plt




class Device:
    def __init__(self,device_id,delay_times=0,peak_voltage=0,voltage_error=0.03,freq=0,data_tensor=[]):
        self.device_id = device_id
        self.delay_times =delay_times 
        self.peak_voltage = peak_voltage
        self.voltage_error = voltage_error
        self.freq = freq
        self.data_tensor=data_tensor

    def set_attribution(self,RH,delay_times):
        self.delay_times = delay_times
        if(RH==90):
            self.peak_voltage = 0.4
            self.freq = 50
        elif(RH==70):
            self.peak_voltage = 0.3
            self.freq = 20
        elif(RH==50):
            self.peak_voltage = 0.2
            self.freq = 10
        elif(RH==30):
            self.peak_voltage = 0.0
            self.freq = 0
        

d1 = Device(device_id=1)
d2 = Device(device_id=2)
d3 = Device(device_id=3)
d4 = Device(device_id=4)
d5 = Device(device_id=5)
d6 = Device(device_id=6)
d7 = Device(device_id=7)
d8 = Device(device_id=8)
d9 = Device(device_id=9)
wind_direction = 4
device_list = [d1,d2,d3,d4,d5,d6,d7,d8,d9]

file_path = 'C:\\Users\\74791\\Desktop\\snntorch-master\\mytest\\dataset\\train\\'

generate_number = 3


#产生一帧数据逻辑
##参数：delay_times:延时时间  peak_voltage:峰值电压  pulse_frequence:脉冲频率
def dataset_generate_one(save_path,filename):
    #命名
    save_path_for_one_data_temp = save_path

    #第一个文件名
    # 文件路径需要处理
    #first_file_2902_path_to_change = first_excel_files_2902.replace("/", "\\")
    #first_save_2902_file_path,first_file_2902_name_with_extension = os.path.split(first_file_2902_path_to_change)
    #first_file_2902_name,first_extension = os.path.splitext(first_file_2902_name_with_extension)
    
    save_path_for_one_data = save_path_for_one_data_temp+filename+'.xls'

    #打开新的表文件，并添加一个sheet
    write_temp_file = xlwt.Workbook(save_path_for_one_data) 
    write_temp_sheet = write_temp_file.add_sheet('Sheet1')

    #接下来开始生成数据
    #写入数据
    group_index = 0
    #一共500个点
    for device_num in range(1,10):
        #选择当前需要处理的器件,每一批器件的数据都要重置
        current_device = device_list[device_num-1]
        time_data_to_write_in_file = []
        voltage_data_to_write_in_file = []
        #写入表头
        write_temp_sheet.write(0,group_index+1,'device'+ str(device_num))

        for data_cnt in range(0, 500):
            time_data_to_write_in_file.append(data_cnt)
            #当实际时间小于延时的时候，不fire
            if(current_device.delay_times>data_cnt):
                voltage_data_to_write_in_file.append(random.uniform(0.001, 0.002))  # 生成 [0.001, 0.002) 之间的随机数
            #否则按概率来产生脉冲
            else:
                if(random.random() < (current_device.freq/(500-current_device.delay_times))):
                    voltage_data_to_write_in_file.append(random.uniform(current_device.peak_voltage-current_device.voltage_error,
                                                                        current_device.peak_voltage+current_device.voltage_error)) 
                else:
                    voltage_data_to_write_in_file.append(random.uniform(0.001, 0.002)) 
                
        current_device.data_tensor=voltage_data_to_write_in_file
        #print(voltage_data_to_write_in_file[300])
        #循环结束就可以开始写入了
        #写入当前组的时间数据
        w_data_row = 1
        for temp_time in time_data_to_write_in_file: 
            write_temp_sheet.write(w_data_row, group_index,temp_time)
            w_data_row = w_data_row + 1

        #写入当前组的电压数据
        w_data_row = 1
        for temp_voltage in voltage_data_to_write_in_file: 
            write_temp_sheet.write(w_data_row, group_index+1, temp_voltage)
            w_data_row = w_data_row + 1
        
        #写入完毕后自增到下一组
        group_index = group_index+2

    #保存文件
    write_temp_file.save(save_path_for_one_data)
    #打印信息
    #print("success")
    #text_box.delete('1.0', tk.END)  # 清空文本框内容
    #text_box.insert(tk.END, '成功')  # 插入文件内容
    #else:
    #打印信息
    #text_box.delete('1.0', tk.END)  # 清空文本框内容
    #text_box.insert(tk.END, '未选择文件')  # 插入文件内容



#一帧数据设置
#321   123
#4 0   456
#567   789
def dataset_attribution_setting(wind_direction,velocity=0):
    if(wind_direction==0):
        d3.set_attribution(RH=90,delay_times=20)
        d6.set_attribution(RH=90,delay_times=20) 
        d9.set_attribution(RH=90,delay_times=20)
        d2.set_attribution(RH=70,delay_times=70)
        d5.set_attribution(RH=70,delay_times=70)
        d8.set_attribution(RH=70,delay_times=70)
        d1.set_attribution(RH=50,delay_times=120)
        d4.set_attribution(RH=50,delay_times=120)
        d7.set_attribution(RH=50,delay_times=120)
    elif(wind_direction==1):
        d1.set_attribution(RH=70,delay_times=120)
        d2.set_attribution(RH=90,delay_times=70) 
        d3.set_attribution(RH=90,delay_times=20)
        d4.set_attribution(RH=50,delay_times=170)
        d5.set_attribution(RH=70,delay_times=120)
        d6.set_attribution(RH=90,delay_times=70)
        d7.set_attribution(RH=30,delay_times=220)
        d8.set_attribution(RH=50,delay_times=170)
        d9.set_attribution(RH=70,delay_times=120)
    elif(wind_direction==2):
        d1.set_attribution(RH=90,delay_times=20)
        d2.set_attribution(RH=90,delay_times=20) 
        d3.set_attribution(RH=90,delay_times=20)
        d4.set_attribution(RH=70,delay_times=70)
        d5.set_attribution(RH=70,delay_times=70)
        d6.set_attribution(RH=70,delay_times=70)
        d7.set_attribution(RH=50,delay_times=120)
        d8.set_attribution(RH=50,delay_times=120)
        d9.set_attribution(RH=50,delay_times=120)
    elif(wind_direction==3):
        d3.set_attribution(RH=70,delay_times=120)
        d2.set_attribution(RH=90,delay_times=70) 
        d1.set_attribution(RH=90,delay_times=20)
        d6.set_attribution(RH=50,delay_times=170)
        d5.set_attribution(RH=70,delay_times=120)
        d4.set_attribution(RH=90,delay_times=70)
        d9.set_attribution(RH=30,delay_times=220)
        d8.set_attribution(RH=50,delay_times=170)
        d7.set_attribution(RH=70,delay_times=120)
    elif(wind_direction==4):
        d1.set_attribution(RH=90,delay_times=20)
        d4.set_attribution(RH=90,delay_times=20) 
        d7.set_attribution(RH=90,delay_times=20)
        d2.set_attribution(RH=70,delay_times=70)
        d5.set_attribution(RH=70,delay_times=70)
        d8.set_attribution(RH=70,delay_times=70)
        d3.set_attribution(RH=50,delay_times=120)
        d6.set_attribution(RH=50,delay_times=120)
        d9.set_attribution(RH=50,delay_times=120)
    elif(wind_direction==5):
        d9.set_attribution(RH=70,delay_times=120)
        d8.set_attribution(RH=90,delay_times=70) 
        d7.set_attribution(RH=90,delay_times=20)
        d6.set_attribution(RH=50,delay_times=170)
        d5.set_attribution(RH=70,delay_times=120)
        d4.set_attribution(RH=90,delay_times=70)
        d3.set_attribution(RH=30,delay_times=220)
        d2.set_attribution(RH=50,delay_times=170)
        d1.set_attribution(RH=70,delay_times=120)
    elif(wind_direction==6):
        d9.set_attribution(RH=90,delay_times=20)
        d8.set_attribution(RH=90,delay_times=20) 
        d7.set_attribution(RH=90,delay_times=20)
        d6.set_attribution(RH=70,delay_times=70)
        d5.set_attribution(RH=70,delay_times=70)
        d4.set_attribution(RH=70,delay_times=70)
        d3.set_attribution(RH=50,delay_times=120)
        d2.set_attribution(RH=50,delay_times=120)
        d1.set_attribution(RH=50,delay_times=120)
    elif(wind_direction==7):
        d7.set_attribution(RH=70,delay_times=120)
        d8.set_attribution(RH=90,delay_times=70) 
        d9.set_attribution(RH=90,delay_times=20)
        d4.set_attribution(RH=50,delay_times=170)
        d5.set_attribution(RH=70,delay_times=120)
        d6.set_attribution(RH=90,delay_times=70)
        d1.set_attribution(RH=30,delay_times=220)
        d2.set_attribution(RH=50,delay_times=170)
        d3.set_attribution(RH=70,delay_times=120)



def dataset_generate(wind_direction_t,save_path,filename):
    #设置好本次要产生的一帧数据的性质
    dataset_attribution_setting(wind_direction=wind_direction_t)
    #产生逻辑
    dataset_generate_one(save_path=save_path,filename=filename)
    

def paint_the_picture():
    plt.figure(figsize=(12, 8))

    time_list = []
    for t in range(0, 500):
        time_list.append(t)

    for i in range(0,9):
        plt.subplot(3, 3, i+1)  # 3行3列布局
        plt.plot(time_list, device_list[i].data_tensor, color='red')
        plt.title(f'Curve {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.xlim(0, 500)   # x轴范围：2到8
        plt.ylim(0, 0.5)  # y轴范围：-1.5到1.5
        plt.grid(True)

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

def dataset_generate_more(wind_direction_t,save_path,num):
    for i in range(0,num):
        dataset_generate(wind_direction_t,file_path,str(wind_direction_t)+'_'+str(i))
        print("success to generate"+str(i))

    paint_the_picture()

dataset_generate_more(7,file_path,generate_number)

