''''
@Project: AI-MSF-benchmark
@Description: Please add Description       
@Time:2022/9/6 15:11       
@Author:NianGao    
 
'''
import os


def add_python_path(p):
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = p
    else:
        os.environ["PYTHONPATH"] += ':{}'.format(p)


def remove_python_path(p):
    p = ':{}'.format(p)
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace(p, "")


def get_op(op):
    if op is None or op == "":
        return "clean"
    else:
        return op


# creat symlink instead of copy files
def symlink(input_path, output_path):
    if not os.path.exists(input_path):
        raise ValueError(input_path)

    if os.path.exists(output_path):
        os.remove(output_path)  # can delet soft link files
    if str(input_path).startswith("./"):
        input_path = os.path.abspath(input_path)
        # convert relateive to abslute
    print(input_path, "-->", output_path)
    os.symlink(input_path, output_path)
