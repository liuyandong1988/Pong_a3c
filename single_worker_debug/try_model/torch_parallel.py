import torch

import torch.multiprocessing as mp


def sub_processes(A, B, D, i, j, size):

    D[(j * size):((j + 1) * size), i] = torch.mul(B[:, i], A[j, i])


def task(A, B):
    size1 = A.shape
    size2 = B.shape
    # 9*3 memory  share CUDA tensors
    D = torch.zeros([size1[0] * size2[0], size1[1]]).cuda()
    D.share_memory_()

    for i in range(1):
        processes = []
        for j in range(size1[0]):
            p = mp.Process(target=sub_processes, args=(A, B, D, i, j, size2[0]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    return D

if __name__ == '__main__':

    A = torch.rand(3, 3).cuda()
    B = torch.rand(3, 3).cuda()
    C = task(A,B)
    torch.multiprocessing.set_start_method("spawn")
    print(C)