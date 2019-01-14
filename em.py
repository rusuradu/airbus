import random
from utility import pgk_xt
from timeit import default_timer as timer


class EM:
    def __init__(self, img, exec_no):
        self.max_step = 12
        self.clusters = 2
        self.miu = [0.0] * self.clusters

        self.miu[0] = random.uniform(0.1, 0.5)
        self.miu[1] = random.uniform(0.5, 0.99)

        self.sigma = [0.0] * self.clusters
        for i in range(self.clusters):
            self.sigma[i] = random.uniform(0.1, 1.0)
        self.pgk = [0.0] * self.clusters
        self.pgk[0] = 0.9
        self.pgk[1] = 0.1
        self.img = img
        self.exec_no = exec_no

    def run_it(self):
        start = timer()
        step = 1
        while step < self.max_step:
            sum_pgkxt = [0.0] * self.clusters
            miu_est = [0.0] * self.clusters
            sigma_est = [0.0] * self.clusters
            total = 0

            for pixel_line in self.img:
                for pixel in pixel_line:
                    for i in range(self.clusters):
                        val = pgk_xt(pixel, self.miu, self.sigma, self.pgk, i, self.clusters)
                        sum_pgkxt[i] = sum_pgkxt[i] + val
                        miu_est[i] = miu_est[i] + val * pixel
                        sigma_est[i] = sigma_est[i] + val * ((pixel - self.miu[i]) ** 2)
                    total = total + 1

            for i in range(self.clusters):
                miu_est[i] = miu_est[i] / sum_pgkxt[i]
                sigma_est[i] = sigma_est[i] / sum_pgkxt[i]
                sum_pgkxt[i] = sum_pgkxt[i] / total

            self.miu = miu_est
            self.sigma = sigma_est
            self.pgk = sum_pgkxt

            step = step + 1
            print("Thread %d, step %d " % (self.exec_no, step - 1) + str(self.pgk))
        end = timer()
        print(end - start)
