function Generate_GA(A, m, n, s; block_sz = 128)

    # Initialize GA
    GA = zeros(s, n)

    # Loop over blocks
    for i in 1:ceil(Int, s / block_sz)
        block_begin = (i - 1) * block_sz + 1
        block_end = min(i * block_sz, s)
        block_len = block_end - block_begin + 1

        # Generate random matrix
        G = randn(block_len, m)

        # Perform matrix multiplication
        GA[block_begin:block_end, :] = G * A
    end

    return GA
end