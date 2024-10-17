运行TPBench
======================

TPBench命令的基本格式
----------------------

    .. code-block:: bash

        -d, --data_dir[=PATH]           Optional. Data directory.
        -g, --group=group_list          Group list. (e.g. -g d_stream).
        -k, --kernel=kernel_list        Kernel list.(e.g. -k d_init,d_sum).
        -L, --list                      List all group and kernels then exit.
        -n, --ntest=number of test      Overall number of tests.
        -s, --nkib=kib_size             Memory usage for a single test array, in KiB.
        -?, --help                      Give this help list.
        -V, --version                   Print program version.
        --usage                         Give a short usage message.


TPBench支持的kernel
----------------------

=======  ============   =============   ===========     ============    ===========
ID       Kernel name    Function name   Bytes/Step      FLOPs           Description
=======  ============   =============   ===========     ============    ===========
1        init           d_init          8               0               FP64 init.
2        sum            d_sum           8               1               FP64 sum.
3        copy           d_copy          16              0               FP64 copy
4        update         d_update        16              1               FP64 update
5        triad          d_triad         24              2               FP64 STREAM Triad
6        axpy           d_axpy          24              2               FP64 AXPY
7        striad	        d_striad        32	            2	            FP64 Stanza Triad.
8        staxpy	        d_staxpy        32	            2	            FP64 Stanza AXPY
9        scale	        d_scale	        16	            1	            FP64 STREAM scale.
10       tl_cgw	        d_tl_cgw        88	            13	            The TeaLeaf cg_calc_w kernel
11       jacobi2d5p     d_jacobi5p      48	            6	            5-point jacobi stencil kernel
12       mulldr	        d_mulldr	    8	            variable	    Configurable mul + load kernel.
13       fmaldr	        d_fmaldr	    8	            variable	    Configurable fma + load kernel.
=======  ============   =============   ===========     ============    ===========


+-----------------+------------+-------------------+-------------------------------------------+
| System          | Kernel     | SIMD              | Description                               |
+=================+============+===================+===========================================+
| Memory          | init       | SVE, AVX512       | a[i]=s                                    |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | sum        | SVE, AVX512       | sum(a[:])                                 |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | copy       | SVE, AVX512       | b[:]=a[:]                                 |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | update     | SVE, AVX512       | a[i]*=s                                   |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | triad      | SVE, AVX512       | a[i]=b[i] + s * c[i]                      |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | axpy       | SVE, AVX512       | a[i]=a[i]+s*b[i]                          |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | striad     | SVE, AVX512       | Strided Triad                             |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | staxpy     | SVE, AVX512       | Strided axpy                              |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | scale      | SVE, AVX512       | a[i]=s*b[i]                               |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | tl_cgw     | N/A               | cg_calc_w stencil in TeaLeaf              |
+-----------------+------------+-------------------+-------------------------------------------+
| Memory          | jacobi2d5p | N/A               | Jacobi 2D 5 point stencil                 |
+-----------------+------------+-------------------+-------------------------------------------+
| Computation     | mulldr     | SVE               | Configurable mul + load kernel.           |
+-----------------+------------+-------------------+-------------------------------------------+
| Computation     | fmaldr     | SVE               | Configurable fma + load kernel.           |
+-----------------+------------+-------------------+-------------------------------------------+
| Communication   | gemm-bcast | SVE, SME, AVX512  | GEMM+Broadcast kernel                     |
+-----------------+------------+-------------------+-------------------------------------------+
| Communication   | Jacobi2d5p-sendrecv | N/A       | jacobi2d5p+Sendrecv kernel               |
+-----------------+------------+-------------------+-------------------------------------------+
| Communication   | gemm-allreduce | SVE, SME, AVX512 | GEMM+Allreduce kernel                  |
+-----------------+------------+-------------------+-------------------------------------------+
