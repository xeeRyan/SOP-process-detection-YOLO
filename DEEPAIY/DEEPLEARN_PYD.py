import json
import multiprocessing
import os, sys
import socket


# share_memery import enforce_train
# import functools
global g_serPort
g_serPort = 5000
def read_json(path):
    # 打开文件,r是读取,encoding是指定编码格式
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    fp.close()
    return data


def save_json(json_data, path):
    with open(path, 'w') as f:
        json.dump(json_data, f)

saveFilePAth = "F:\\Test\\command_line_args.txt"
def save_string_to_file_basic(text, filename):
    """基本保存方法，如果文件存在则追加，否则创建"""
    # 检查文件是否存在
    if os.path.exists(filename):
        mode = 'a'  # 追加模式
        print(f"文件已存在，将以追加模式写入: {filename}")
    else:
        mode = 'w'  # 写入模式（创建新文件）
        print(f"文件不存在，将创建新文件: {filename}")

    with open(filename, mode, encoding='utf-8') as file:
        file.writelines(str(text + "\n"))

    print(f"字符串已保存到: {filename}")


def handle_client(client, client_addr):
    # save_string_to_file_basic("handle", saveFilePAth)
    BUFSIZE = 102400  # 传输字节数
    while True:
        try:
            json_b = client.recv(BUFSIZE)
            print("recv", json_b)
            if not json_b or json_b == b'end':
                print("close recv")
                client.close()
                break
            elif json_b == b'close':
                client.close()
                return "close"
        except Exception as e:
            print("except disconnect")
            # save_string_to_file_basic(f"except:{e}", "F:\\Test\\except.txt")
            break

        try:

            decoded_str = json_b.decode('utf-8')
            # 提取路径部分（去掉开头的'model_file:'）
            json_ss = decoded_str.split(':', 1)[1]

            # save_string_to_file_basic(json_ss, saveFilePAth)

            # json_s = json_b.decode("utf-8").encode()
            #
            # json_ss = json_s.split("model_file:", 1)[1]  # 分割一次，取第二部分
            dict_data = read_json(json_ss)
            # dict_data = json.dumps(strJson, indent=4, ensure_ascii=False)  # 其中ensure_ascii确保中文能够显示
            # dict_data = json.loads(json_s)
            print(dict_data)
            print("enter train")
            from ModelSelection import SelectModel
            data = SelectModel(dict_data)
            print("train end")
            client.sendto(b'train normal quit', client_addr)
            print("return tcp msg")
        except:
            print("error train")
            print("end rec:", json_b)
            client.sendto(b'train except quit', client_addr)
            print("return tcp except msg")
    return ""


def Start_tcp(cmd='open_tcp'):
    ip_port = ('127.0.0.1', g_serPort)
    # save_string_to_file_basic("Start_tcp", saveFilePAth)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # 这个设置是上一个服务器未正常关闭也能使用 相同的地址和端口。
    s.bind(ip_port)
    s.listen(1)
    while True:
        client, client_addr = s.accept()
        strCmd = ''
        print("listen a client")
        # save_string_to_file_basic("listen a client", saveFilePAth)
        strCmd = handle_client(client, client_addr)

        if len(strCmd) > 0 and strCmd == 'close':
            break
        print('continue')
    s.close()


def get_instance(dict_data):
    print('start import torch')
    from ModelSelection import SelectModel
    print('end import torch')
    print('enter dll')
    data = SelectModel(dict_data)  # dict
    print('quit dll')


def train_test(json_path):
    strJson = read_json(json_path)
    json_data = json.dumps(strJson, indent=4, ensure_ascii=False)   # 其中ensure_ascii确保中文能够显示
    print(json_data)
    get_instance(strJson)


def train_test2():
    list = sys.argv
    print(list)
    length = len(list)
    # save_string_to_file_basic(str(length), saveFilePAth)
    if length == 2:
        # json_s = list[1].decode("utf-8").encode()
        # json_ss = json_s.split("model_file:", 1)[1]  # 分割一次，取第二部分
        # save_string_to_file_basic(json_ss, "F:\\Test\\1.txt")
        # strJson = read_json(json_ss)
        # dict_data = json.dumps(strJson, indent=4, ensure_ascii=False)  # 其中ensure_ascii确保中文能够显示

        dict_data = json.loads(list[1])
        sys.argv.remove(list[1])
        get_instance(dict_data)
    else:
        Start_tcp()

# def just_one_instance(func):
#     @functools.wraps(func)
#     def f(*args, **kwargs):
#         try:
#             global ws
#             ws = socket.socket()
#             host = socket.gethostname()
#             ws.bind((host, 60001))
#         except:
#             print("is open")
#             return None
#         return func(*args, **kwargs)
#     return f


# @just_one_instance
def main_instance():
    print("main start")

    list = sys.argv
    arg_count = len(list)

    # with open('F:\\Test\\command_line_args.txt', 'w', encoding='utf-8') as f:
    #     f.write("命令行参数列表:\n")
    #     f.write("=" * 40 + "\n")
    #     for i, arg in enumerate(list):
    #         f.write(f"[{i}] {arg}\n")



    if arg_count == 1:
        return

    if list[1] == "tcp":
        global g_serPort
        g_serPort = 5000 if arg_count < 3 else int(list[2])
        sys.argv.remove("tcp")
        sys.argv.remove(str(g_serPort))

        # save_string_to_file_basic("htrain_test2", saveFilePAth)
        train_test2()

    elif list[1] == "load_model" and arg_count == 3:
        # json_s = list[2].decode("utf-8").encode()
        # json_ss = json_s.split("model_file:", 1)[1]  # 分割一次，取第二部分
        # strJson = read_json(json_ss)
        # dict_data = json.dumps(strJson, indent=4, ensure_ascii=False)  # 其中ensure_ascii确保中文能够显示
        dict_data = json.loads(list[2])
        # call_func(dict_data)
        from ModelSelection import parse_model
        parse_model(dict_data)
    elif list[1] == "load_model_file" and arg_count == 3:
        json_s = list[2].decode("utf-8").encode()
        # json_ss = json_s.split("model_file:", 1)[1]  # 分割一次，取第二部分
        strJson = read_json(json_s)
        dict_data = json.dumps(strJson, indent=4, ensure_ascii=False)  # 其中ensure_ascii确保中文能够显示
        # dict_data = json.loads(list[2])
        # call_func(dict_data)
        from ModelSelection import parse_model
        parse_model(dict_data)
    else:
        return


def local_test():
    list = sys.argv
    arg_count = len(list)

    print("args1:", sys.argv)
    current_path = os.path.dirname(sys.argv[0])
    print(current_path)
    if arg_count == 1:
        return

    path = list[1]
    sys.argv.remove(path)

    print("args2:", sys.argv[0])
    current_path = os.path.dirname(sys.argv[0])
    print(current_path)

    data = read_json(path)
    get_instance(data)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # AIPRO运行接口
    # main_instance()

    # train_test2()
    # print("start exe")

    # local_test()
    # train_test()
    """ocr"""
    # # train
    # json_path = "E:\\BSW\\BSW_project\\project_OCR\\0307-OCR\\temp\\qt_2023_11_09_10_14_02.json"
    # json_path = "E:\\BSW\\project_OCR\\0420-利捷得-字符识别\\temp\\qt_2023_04_21_09_25_50.json"
    # # val
    # json_path = "E:\\BSW\\BSW_project_m\\1212-OCR\\temp\\2expand.bmp.json"
    # json_path = "E:\\BSW\\project_OCR\\test\\0322-OCR\\temp\\8_20230103-183054551.png.json"
    # # test
    # json_path = "E:\\BSW\\project_OCR\\0314-OCR\\temp\\0_test.json"
    """location"""
    # train
    # json_path = "E:\\task\\June\\2024_6_15\\0411-尿液L\\temp\\qt_2024_06_15_14_31_33.json"
    # 大模型
    # json_path = "E:\\BSW\\BSW_project_m\\1215-定位-OCR\\temp\\qt_2023_05_06_16_29_10.json"
    # valid
    # json_path = "E:\\BSW\\BSW_project_m\\1215-定位-OCR\\temp\\15expand.bmp.json"
    # test
    # json_path = "E:\\BSW\\BSW_project\\project_Loc\\0411-尿液L\\temp\\0_test.json"

    """Segment"""
    # json_path = input("请输入json文件路径：")
    # json_path = "F:\\项目\\项目测试库\\正样本\\7\\temp\\qt_2026_04_21_15_49_33.json"
    # json_path = "D:\\BSW\\aipro_exe\\json_cache\\2026-01-24\\2_20260124174526fff.json"
    # #json_path = "E:\\task\\June\\2024_6_15\\0411-尿液L\\temp\\qt_2025_02_14_10_04_19.json"
    json_path = "G:\\项目工程\\AIPro\\切图\\切图\\temp\\qt_2026_05_26_11_37_09.json"
    train_test(json_path)
    # print("完成训练")