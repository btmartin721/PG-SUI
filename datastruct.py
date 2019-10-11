
class Struct:

    def __init__(self, id, loci):
        self.id = id
        self.loci = loci

    def make_dict(self, num, mydict, pattrn):
        mydict[pattrn] = num
