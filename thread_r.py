from multiprocessing import Process


class my_process(Process):

    def __init__(self, method, img_ids, index):
        super(my_process, self).__init__()
        self.method = method
        self.img_ids = img_ids
        self.index = index

    def run(self):
        ind = 0
        mx = len(self.img_ids)
        for img_id in self.img_ids:
            print("Thread %d: Process %d out of %d" % (self.index, ind, mx))
            self.method(img_id, self.index)
            ind = ind + 1
