TPBench的内核设计
============================

评测内存子系统的计算内核设计
----------------------------

对于评估内存子系统的计算内核，TPBench报告的性能指标为数据读写速率，具体有两个指标：单个时钟周期内的读写数据量B/c(Bytes/cycle)和数据读写带宽MB/s。其计算方式为：
    
        Bytes_per_cycle = bytes_per_step * num_steps / elapsed_cycles
    
        MB_per_seconds = bytes_per_step * num_steps / elapsed_seconds / 10^6.

其中bytes_per_step由计算内核类型决定；num_steps为一位数组的长度或二维矩阵的元素个数，elapsed_cycles为计时得到的消耗的CPU时钟周期数目，elapsed_seconds为计时得到的消耗的墙钟时间。

TPBench会统计并报告ntests个性能数据的均值、最大值、最小值、25%分位值、中位值和75%分位值。


init
~~~~~~~~~~~~~

    - 计算过程
        将数组a的所有元素全部初始化为s。a[i]=s, i in 1…n。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次基于SVE指令的init计算。
        
        .. code-block:: C
            
            for(int i = 0; i < ntest; i ++){
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, s, nsize)
                for (int j = 0; j < nsize; j += vec_len) {
                    svbool_t predicate = svwhilelt_b64_s32(j, nsize);
                    svfloat64_t vec = svdup_f64(s);
                    svst1_f64(predicate, a + j, vec);
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a占用内存空间的大小，单位为KiB。
    - 数据初始化方式
        数组a不进行初始化，标量s初始化为0.99。


sum
~~~~~~~~~~~~~

    - 计算过程
        对数组a的所有元素进行求和。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次基于SVE指令的sum计算。
        
        .. code-block:: C
            
            for(int i = 0; i < ntest; i ++){
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                svfloat64_t vec_s = svdup_f64(0);
                #pragma omp parallel for shared(a,nsize) reduction(+:s)
                for(int j = 0; j < nsize; j += vec_len){
                    svbool_t predicate = svwhilelt_b64_s32(j, nsize);
                    svfloat64_t vec_a = svld1_f64(predicate, a + j);
                    vec_s = svadd_f64_z(predicate, vec_s, vec_a);
                }
                s += svaddv(svptrue_b64(), vec_s);
                __getcy_1d_en(i);
                __getns_1d_en(i);
                // Reset s to 0.11
                s = s / (double)(nsize + 1);
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a占用内存空间的大小，单位为KiB。
    - 数据初始化方式
        数组a中的每个元素初始化为0.11，标量s初始化为0.11。



copy
~~~~~~~~~~~~~

    - 计算过程
        将数组b的内容复制到数组a中。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次基于SVE指令的init计算。
        
        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, b, c, s, nsize)
                for (int j = 0; j < nsize; j += vec_len) {
                    svbool_t predicate = svwhilelt_b64_s32(j, nsize);
                    svfloat64_t vec_b = svld1_f64(predicate, b + j);
                    svst1_f64(predicate, a + j, vec_b);
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }


    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a（或b）占用内存空间的大小，单位为KiB。
    - 数据初始化方式
        a[i]=0.11, b[i]=0.11 + i, i=1…n。


update
~~~~~~~~~~~~~

    - 计算过程
        对数组a进行逐元素与标量s相乘的计算：a[i] = s * a[i], i=1…n。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次基于SVE指令的计算内核的运算。
        
        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, s, nsize)
                for (int j = 0; j < nsize; j += vec_len) {
                    svbool_t predicate = svwhilelt_b64_s32(j, nsize);
                    svfloat64_t vec_a = svld1_f64(predicate, a + j);
                    vec_a = svmul_n_f64_z(predicate, vec_a, s);
                    svst1_f64(predicate, a + j, vec_a);
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }



    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a占用内存空间的大小，单位为KiB。
    - 数据初始化方式
        a[i]=0.9999, i=1…n。
        s=0.9999。

triad
~~~~~~~~~~~~~

    - 计算过程
        对三个数组a,b,c进行如下计算：a[i]=b[i] + s * c[i]，i=1…n。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次基于SVE指令的计算内核的运算。
        
        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, b, c, s, nsize)
                for (int j = 0; j < nsize; j += vec_len) {
                    svbool_t predicate = svwhilelt_b64_s32(j, nsize);
                    svfloat64_t vec_b = svld1_f64(predicate, b + j);
                    svfloat64_t vec_c = svld1_f64(predicate, c + j);
                    svfloat64_t vec_a = svmla_n_f64_z(predicate,vec_b,vec_c, s);
                    svst1_f64(predicate, a + j, vec_a);
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a（或b、c）占用内存空间的大小，单位为KiB。
    - 数据初始化方式
        a[i]=1.0, b[i]=2.0, c[i]=3.0, i=1…n
        s=0.42

axpy
~~~~~~~~~~~~~

    - 计算过程
        对数组a,b逐元素进行如下计算：a[i] = a[i] + s * b[i]，i=1…n。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次计算内核的运算。
        
        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, b, s, nsize)
                for (int j = 0; j < nsize; j ++) {
                    svbool_t predicate = svwhilelt_b64_s32(j, nsize);
                    svfloat64_t vec_a = svld1_f64(predicate, a + j);
                    svfloat64_t vec_b = svld1_f64(predicate, b + j);
                    vec_a = svmla_n_f64_z(predicate, vec_a, vec_b, s);
                    svst1_f64(predicate, a + j, vec_a);
                }
                #pragma omp barrier
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }


    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a（或b）占用内存空间的大小，单位为KiB。
    - 数据初始化方式
        a[i]=0.11, b[i]=0.11, i=1…n
        s=0.11


striad
~~~~~~~~~~~~~

    - 计算过程
        对数组a,b,c进行有stride的triad计算：a[i] = b[i] + s * c[i], if i mod (stride + L) < stride, i=1…n。 
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次计算内核的运算。
        
        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, b, c, s, stride, nsize, jump, nb)
                for (int j = 0; j < nb; j ++) {
                    for (int k = j * jump; k < j * jump + stride; k += vec_len) {
                        svbool_t predicate = svwhilelt_b64_s32(k, j * jump + stride);
                        svfloat64_t vec_b = svld1_f64(predicate, b + k);
                        svfloat64_t vec_c = svld1_f64(predicate, c + k);
                        svfloat64_t vec_a = svmla_n_f64_z(predicate,vec_b,vec_c, s);
                        svst1_f64(predicate, a + k, vec_a);
                    }
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a（或b、c）占用内存空间的大小，单位为KiB。
        - stride: 默认为8，可通过设置环境变量TPBENCH_STRIDE进行调节。
        - L：默认为8，可通过设置环境变量TPBENCH_L进行调节。

    - 数据初始化方式
        a[i]=0.11, b[i]=0.11, i=1…n
        s=0.11

staxpy
~~~~~~~~~~~~~

    - 计算过程
        对数组a,b,c进行有stride的axpy计算：a[i] = a[i] + s * b[i], if i mod (stride + L) < stride, i=1…n。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次计算内核的运算。
        
        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, b, s, stride, nsize, jump, nb)
                for (int j = 0; j < nb; j ++) {
                    for (int k = j * jump; k < j * jump + stride; k += vec_len) {
                        svbool_t predicate = svwhilelt_b64_s32(k, j * jump + stride);
                        svfloat64_t vec_a = svld1_f64(predicate, a + k);
                        svfloat64_t vec_b = svld1_f64(predicate, b + k);
                        vec_a = svmla_n_f64_z(predicate, vec_a, vec_b, s);
                        svst1_f64(predicate, a + k, vec_a);
                    }
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a（或b、c）占用内存空间的大小，单位为KiB。
        - stride: 默认为8，可通过设置环境变量TPBENCH_STRIDE进行调节。
        - L：默认为8，可通过设置环境变量TPBENCH_L进行调节。

    - 数据初始化方式
        a[i]=0.11, b[i]=0.11, i=1…n
        s=0.11

scale
~~~~~~~~~~~~~

    - 计算过程
        对数组a,b逐元素进行如下计算：a[i]=s*b[i]，i=1…n。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次计算内核的运算。
        
        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                #pragma omp parallel for shared(a, b, c, s, nsize)
                for (int j = 0; j < nsize; j += vec_len) {
                    svbool_t predicate = svwhilelt_b64_s32(j, nsize);
                    svfloat64_t vec_b = svld1_f64(predicate, b + j);
                    svfloat64_t vec_a = svmul_n_f64_z(predicate, vec_b, s);
                    svst1_f64(predicate, a + j, vec_a);
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a（或b、c）占用内存空间的大小，单位为KiB。

    - 数据初始化方式
        a[i]=0.11, b[i]=0.11, i=1…n
        s=0.11

tl_cgw
~~~~~~~~~~~~~

    - 计算过程
        从TeaLeaf迷你应用中提取出的cg_calc_w计算内核：对二维矩阵w,Di,p,Kx,Ky进行5 point stencil计算，并计算w和p的内积。同时TPBench实现了分块的版本。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次计算内核的运算。

        .. code-block:: C
            
            static void run_kernel_once(int narr) {
                pw = 0;
                if (block_x == 0) { 
                    // no blocking
                    for (int i = 1; i < height-1; i ++) {
                        for (int j = 1; j < narr-1; j ++) {
                            w[i][j] = Di[i][j] * p[i][j]  \
                                        - ry * (Ky[i+1][j] * p[i+1][j] + Ky[i][j] * p[i-1][j]) \
                                        - rx * (Kx[i][j+1] * p[i][j+1] + Kx[i][j] * p[i][j-1]);
                        }
                        for (int j = 1; j < narr-1; j ++) {
                            pw = pw + w[i][j] * p[i][j];
                        }
                    }
                }
                // blocking
                else {
                    for (int bx = 0; bx < narr - 1; bx += block_x) {
                        for (int i = 1; i < height - 1; i++) {
                            for (int j = MAX(bx, 1); j < MIN(bx + block_x, narr - 1); j++) {
                                w[i][j] = Di[i][j] * p[i][j]  \
                                            - ry * (Ky[i+1][j] * p[i+1][j] + Ky[i][j] * p[i-1][j]) \
                                            - rx * (Kx[i][j+1] * p[i][j+1] + Kx[i][j] * p[i][j-1]);
                            }
                            for (int j = MAX(bx, 1); j < MIN(bx + block_x, narr - 1); j++) {
                                pw = pw + w[i][j] * p[i][j];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);

                run_kernel_once(nsize);

                __getcy_1d_en(i);
                __getns_1d_en(i);
            }


    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - nsize: 二维矩阵的长度（矩阵为方阵，长和宽相等）。在代码中此变量复用了kib的名称。
        - block_x：j维分块的大小，默认为0，代表不分块，通过环境变量TPBENCH_BLOCK来进行设置。


    - 数据初始化方式
        5个二维矩阵w,Di,p,Kx和Ky均进行随机初始化。rx=0.11, ry=0.22。


jacobi2d5p
~~~~~~~~~~~~~

    - 计算过程
        对二维矩阵out,in进行5 point stencil计算：out[j][k] = a * in[j][k] + b * (in[j+1][k]+ in[j][k+1]+ in[j-1][k]+ in[j][k-1])。同时TPBench实现了一维分块的版本。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。在每个计时区域内，进行一次计算内核的运算。

        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_1d_st(i);
                __getcy_1d_st(i);
                if (block_size == 0) { 
                    // no blocking
                    for (int j = 1; j < height-1; j ++) {
                        for (int k = 1; k < nsize-1; k ++) {
                            out[j][k] = a * in[j][k] + b * (in[j-1][k] + in[j+1][k] + in[j][k-1] + in[j][k+1]);
                        }
                    }
                }
                // blocking
                else {
                    for (int bx = 1; bx < nsize-1; bx += block_size) {
                        for (int j = 1; j < height-1; j++) {
                            for (int k = bx; k < MIN(bx + block_size, nsize-1); k++) {
                                out[j][k] = a * in[j][k] + b * (in[j-1][k] + in[j+1][k] + in[j][k-1] + in[j][k+1]);
                            }
                        }
                    }
                }
                __getcy_1d_en(i);
                __getns_1d_en(i);
            }



    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - nsize: 二维矩阵的长度（矩阵为方阵，长和宽相等）。在代码中此变量复用了kib的名称。
        - block_size：j维分块的大小，默认为0，代表不分块，通过环境变量TPBENCH_BLOCK来进行设置。

    - 数据初始化方式
        二维矩阵out和in均进行随机初始化，a=0.21, b=0.2。


评测计算子系统的计算内核设计
----------------------------

对于fmaldr和mulldr这两个的计算内核，TPBench报告浮点计算的性能指标，以求与Roofline模型保持一致。具体包括两个指标：单个时钟周期内的flops和单位时间（1s）的flops。其计算方式为：
				
        flops_per_cycle = flops_per_step * num_steps / elapsed_cycles

        Mflops_per_seconds = flops _per_step * num_steps / elapsed_seconds / 10^6.

其中flops_per_step由计算内核的计算访存比决定。

TPBench会统计并报告ntests个性能数据的均值、最大值、最小值、25%分位值、中位值和75%分位值。


mulldr
~~~~~~~~~~~~~

    - 计算过程
        此计算内核用于Roofline模型的建模。该计算内核对数组a逐元素进行顺序load操作，同时根据设置的计算访存比，进行对应数量的MUL指令的计算。
    - 计算内核代码与计时方式
        TPBench针对每种支持的计算访存比大小均提供了对应的计算内核。例如，计算密度为0.125（1个SVE MUL指令对应1个SVE LOAD指令）的内核实现如下：

        .. code-block:: C
            
            static void run_kernel_once(int nsize) {
                svbool_t predicate = svwhilelt_b64_s32(0, 8);
                for (int r = 0; r < repeat; r++) {
                    #pragma GCC unroll 32
                    for (int i = 0; i < nsize; i += 8) {
                        svfloat64_t reg = svld1_f64(predicate, &a[i]);
                        asm volatile (
                            "fmul z1.d, %[reg].d, %[reg].d\n\t"
                            :: [reg] "w" (reg)
                            : "z1"
                        );
                    }
                }
            }
        
        其他配置的内核实现详见文件src/kernels/simple/mulldr.c。

        TPBench重复运行计算内核ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a占用内存空间的大小，单位为KiB。
        - 计算访存比：由于此参数为编译期参数，其修改需要手动注释和取消注释源文件中的宏定义：

        .. code-block:: C
            
            /* change this macro to adjust the Compute:Load ratio */
            // #define FL_RATIO_F1L4 1
            // #define FL_RATIO_F1L2 1
            #define FL_RATIO_F1L1 1
            // #define FL_RATIO_F2L1 1
            // #define FL_RATIO_F3L1 1
            // #define FL_RATIO_F4L1 1
            // #define FL_RATIO_F8L1 1
            // #define FL_RATIO_F16L1 1
            // #define FL_RATIO_F32L1 1

        上面给出了TPBench支持的所有计算访存比：MUL与LOAD指令个数比例为：1:4, 1:2, 1:1, 2:1, 3:1, 4:1, 8:1, 16:1, 32:1。如果想要设置MUL与LOAD比例为1:1，则取消注释FL_RATIO_F1L1宏，注释掉其他所有宏。

    - 数据初始化方式
        a[i]=1.23, i=1…n。



fmaldr
~~~~~~~~~~~~~

    - 计算过程
        此计算内核用于Roofline模型的建模。该计算内核对数组a逐元素进行顺序load操作，同时根据设置的计算访存比，进行对应数量的FMA指令的计算。
    - 计算内核代码与计时方式
        TPBench针对每种支持的计算访存比大小均提供了对应的计算内核。例如，计算密度为0.125（1个SVE FMA指令对应1个SVE LOAD指令）的内核实现如下：

        .. code-block:: C
            
            static void run_kernel_once(int nsize) {
                svbool_t predicate = svwhilelt_b64_s32(0, 8);
                svfloat64_t z0, z1, z2, z3, z4, z5, z6, z7, z8, 
                            w1, w2, w3, w4, w5, w6, w7, w8;

                INIT_REGISTERS
                
                for (int r = 0; r < repeat; r++) {
                    for (int i = 0; i < nsize; i += 128) {
                        z0 = svld1_f64(predicate, &a[i]);
                        z1 = svmad_f64_x(predicate, z1, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 8]);
                        z2 = svmad_f64_x(predicate, z2, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 16]);
                        z3 = svmad_f64_x(predicate, z3, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 24]);
                        z4 = svmad_f64_x(predicate, z4, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 32]);
                        z5 = svmad_f64_x(predicate, z5, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 40]);
                        z6 = svmad_f64_x(predicate, z6, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 48]);
                        z7 = svmad_f64_x(predicate, z7, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 56]);
                        z8 = svmad_f64_x(predicate, z8, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 64]);
                        w1 = svmad_f64_x(predicate, w1, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 72]);
                        w2 = svmad_f64_x(predicate, w2, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 80]);
                        w3 = svmad_f64_x(predicate, w3, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 8 * 11]);
                        w4 = svmad_f64_x(predicate, w4, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 8 * 12]);
                        w5 = svmad_f64_x(predicate, w5, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 8 * 13]);
                        w6 = svmad_f64_x(predicate, w6, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 8 * 14]);
                        w7 = svmad_f64_x(predicate, w7, z0, z0);
                        z0 = svld1_f64(predicate, &a[i + 8 * 15]);
                        w8 = svmad_f64_x(predicate, w8, z0, z0);
                        asm volatile (
                        "\n\t":: "w" (z0), "w" (z1), "w" (z2), "w" (z3), "w" (z4), 
                                    "w" (z5), "w" (z6), "w" (z7), "w" (z8),
                                    "w" (w1), "w" (w2), "w" (w3), "w" (w4), 
                                    "w" (w5), "w" (w6), "w" (w7), "w" (w8):
                        );
                    }
                }
            }

        上述实现对内层循环展开了16次，如此做的原因是FMA指令的输出寄存器同样作为输入寄存器，导致多次运行的输出寄存器相同的FMA指令之间存在着数据依赖，因此需要多个输出寄存器不同的FMA指令来掩盖延时。

        其他配置的内核实现详见文件src/kernels/simple/fmaldr.c。

        TPBench重复运行计算内核ntest次，在每次重复运行的首尾进行计时，记录墙钟时间（__getns_1d_st）和CPU时钟周期数（__getcy_1d_st）。

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - kib: 数组a占用内存空间的大小，单位为KiB。
        - 计算访存比：由于此参数为编译期参数，其修改需要手动注释和取消注释源文件中的宏定义：

        .. code-block:: C
            
            /* change this macro to adjust the Compute:Load ratio */
            // #define FL_RATIO_F1L4 1
            // #define FL_RATIO_F1L2 1
            #define FL_RATIO_F1L1 1
            // #define FL_RATIO_F2L1 1
            // #define FL_RATIO_F4L1 1
            // #define FL_RATIO_F8L1 1


        上面给出了TPBench支持的所有计算访存比：MUL与LOAD指令个数比例为：1:4, 1:2, 1:1, 2:1, 4:1, 8:1。如果想要设置MUL与LOAD比例为1:1，则取消注释FL_RATIO_F1L1宏，注释掉其他所有宏。

    - 数据初始化方式
        a[i]=1.23, i=1…n。


评测通信子系统的计算内核设计
----------------------------



Gemm+Bcast
~~~~~~~~~~~~~

    - 计算过程
        各个进程独立计算NxN方阵的矩阵乘法C=A*B，然后0号进程使用MPI_Bcast广播A中前Nr行的元素。此计算内核模拟真实应用中计算与通信交替进行的场景，评测此情形下的通信性能。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在运行中，会对GEMM和MPI_Bcast通信进行分别计时，记录墙钟时间（__getns_2d_st）和CPU时钟周期数（__getcy_2d_st）。

        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_2d_st(i, 1);
                __getcy_2d_st(i, 1);
                C = gemm(A, B);
                __getcy_2d_en(i, 1);
                __getns_2d_en(i, 1);
                __getns_2d_st(i, 2);
                __getcy_2d_st(i, 2);
                MPI_Bcast(A[:Nr][:], root=0);
                __getcy_2d_en(i, 2);
                __getns_2d_en(i, 2);    
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - N：方阵的维度。
        - Nr：A矩阵进行广播的行数。默认为10，可通过设置环境变量TPBENCH_GEMM_NR来修改。
        - skip_comp：省略GEMM计算，默认为false，不生效。可通过设置环境变量TPBENCH_SKIP_COMP=1来使其生效，此时程序只进行MPI通信。使用此选项可以对比计算和通信重叠与仅有通信两种场景下的通信性能和性能波动情况。
        - GEMM使用的SIMD指令集：默认使用scalar指令；当GCC编译传入-DKP_SVE时，使用SVE指令；传入-DKP_SME时，使用SME指令。

    - 数据初始化方式
        矩阵A和B进行随机初始化。

Gemm+Allreduce
~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 计算过程
        各个进程独立计算NxN方阵的矩阵乘法C=A*B，然后对C中前Nr行的元素进行Allreduce求和，保存到每个进程A[0][0]的位置。此计算内核模拟真实应用中计算与通信交替进行的场景，评测此情形下的通信性能。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在运行中，会对GEMM和MPI_Allreduce通信进行分别计时，记录墙钟时间（__getns_2d_st）和CPU时钟周期数（__getcy_2d_st）。

        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_2d_st(i, 1);
                __getcy_2d_st(i, 1);
                C = gemm(A, B);
                __getcy_2d_en(i, 1);
                __getns_2d_en(i, 1);
                __getns_2d_st(i, 2);
                __getcy_2d_st(i, 2);
                MPI_Allreduce(C[:Nr][:], dst=A[0][0]);
                __getcy_2d_en(i, 2);
                __getns_2d_en(i, 2);
            }


    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - N：方阵的维度。
        - Nr：A矩阵进行广播的行数。默认为10，可通过设置环境变量TPBENCH_GEMM_NR来修改。
        - skip_comp：省略GEMM计算，默认为false，不生效。可通过设置环境变量TPBENCH_SKIP_COMP=1来使其生效，此时程序只进行MPI通信。使用此选项可以对比计算和通信重叠与仅有通信两种场景下的通信性能和性能波动情况。
        - GEMM使用的SIMD指令集：默认使用scalar指令；当GCC编译传入-DKP_SVE时，使用SVE指令；传入-DKP_SME时，使用SME指令。

    - 数据初始化方式
        矩阵A和B进行随机初始化。


Jacobi2d5p+Sendrecv
~~~~~~~~~~~~~~~~~~~~~~~~~~

对于上述计算与通信重叠的内核，TPBench会分别报告计算时间和通信时间。为了在不同进程之间进行对比，TPBench会给出每个进程的计算和通信耗时。

TPBench会统计并报告ntests个性能数据的均值、最大值、最小值、25%分位值、中位值和75%分位值。

    - 计算过程
        Jacobi2d5p stencil的计算加halo exchange。每个进程首先NxN的区域进行Jacobi2d5p stencild 计算，然后使用sendrecv通信原语与上方和下方进程通信，分别交换矩阵边界第一行和最后一行的元素。此计算内核模拟真实应用中计算与通信交替进行的场景，评测此情形下的通信性能。
    - 计算内核代码与计时方式
        计算内核重复运行ntest次，在运行中，会对GEMM和MPI_Bcast通信进行分别计时，记录墙钟时间（__getns_2d_st）和CPU时钟周期数（__getcy_2d_st）。

        .. code-block:: C
            
            for (int i = 0; i < ntest; i ++) {
                tpmpi_dbarrier();
                __getns_2d_st(i, 1);
                __getcy_2d_st(i, 1);
                for (int i = 1; i <= N; i++) {
                    for (int j = 1; j <= N; j++) {
                        B(i, j) = 1.23 * A(i, j) + 1.56 * (A(i-1,j) + A(i+1,j) + A(i,j-1) + A(i,j+1));
                    }
            }
                __getcy_2d_en(i, 1);
                __getns_2d_en(i, 1);
                __getns_2d_st(i, 2);
                __getcy_2d_st(i, 2);
                MPI_Sendrecv(up_rank, src=B[1,1:N+1], dst=A[0, 1:N+1]);
                MPI_Sendrecv(down_rank, src=B[N,1:N+1], dst=A[N+1, 1:N+1]);
                __getcy_2d_en(i, 2);
                __getns_2d_en(i, 2);
            }

    - 输入参数
        - ntest: 计算内核重复运行的次数。
        - N：方阵的维度。
        - skip_comp：省略GEMM计算，默认为false，不生效。可通过设置环境变量TPBENCH_SKIP_COMP=1来使其生效，此时程序只进行MPI通信。使用此选项可以对比计算和通信重叠与仅有通信两种场景下的通信性能和性能波动情况。

    - 数据初始化方式
        Jacobi二维输入矩阵A和输出矩阵B进行随机初始化。

