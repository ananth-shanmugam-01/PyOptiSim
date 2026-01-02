class Smooth:
    @staticmethod
    def max(x, y, eps=1e-3):
        return 0.5*(x + y + ca.sqrt((x - y)**2 + eps**2))

    @staticmethod
    def min(x, y, eps=1e-3):
        return 0.5*(x + y - ca.sqrt((x - y)**2 + eps**2))
 