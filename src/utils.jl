"""
    Generate_GA(A, m, n, s; block_sz = 128)
Generate a random matrix GA = G * A where G is a potentially large random matrix following a normal distribution.
"""

"""
    Generate_AG(A, m, n, s; block_sz = 128)
Generate a random matrix AG = A * G where G is a potentially large random matrix following a normal distribution.
"""

function Generate_GA(A, m, n, s, R; block_sz = 128)

    # Initialize GA
    GA = zeros(R, s, n)

    # Loop over blocks
    for i in 1:ceil(Int, s / block_sz)
        block_begin = (i - 1) * block_sz + 1
        block_end = min(i * block_sz, s)
        block_len = block_end - block_begin + 1

        # Generate random matrix
        G = randn(R,block_len, m)

        # Perform matrix multiplication
        GA[block_begin:block_end, :] = G * A
    end

    return GA
end

function Generate_AG(A, m, n, s, R; block_sz = 128)

    # Initialize GA
    AG = zeros(R, m, s)

    # Loop over blocks
    for i in 1:ceil(Int, s / block_sz)
        block_begin = (i - 1) * block_sz + 1
        block_end = min(i * block_sz, s)
        block_len = block_end - block_begin + 1

        # Generate random matrix
        G = randn(R, n, block_len)

        # Perform matrix multiplication
        AG[:, block_begin:block_end] = A * G
    end

    return AG
end
