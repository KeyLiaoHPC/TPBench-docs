安装与环境配置
================

TPBench依赖
----------------

TPBench需要如下基本的环境及工具：

- Linux操作系统，内核版本不低于3.10。用户需要使用root权限来加载内核模块，用于配置性能计数器。
- GCC编译器，版本不低于8.3。
- GNU Autotools。
- OpenMPI，版本不低于3.10.0。
- GNU GSL (GNU Scientific Library)，版本不低于2.7。
- OpenBLAS，版本不低于0.3.23。

TPBench编译
----------------

1. 解压TPBench.zip，正确编译安装所有依赖软件，并配置环境变量。
2. 进入setup目录，将Make.gcc_armv8复制为Make.<your_name>，修改CFLAGS如下：
    .. code-block:: bash

        CFLAGS=-O3 -DUSE_MPI -fno-builtin -march=armv8-a+sve -mtune=native -DKP_SVE -Wno-discarded-qualifiers -Wno-incompatible-pointer-types

3. 返回根目录，编译TPBench，对于Armv8架构，需要编译并加载内核模块：
    .. code-block:: bash

        cd .. && make SETUP=<your_name>
        su -
        cd pmu && make && insmod ./enable_pmu 

