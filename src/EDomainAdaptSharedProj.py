
class EDACMD:
    none = 0
    da_m_mine1 = 'da_m_mine1'

    @staticmethod
    def is_multi(cmd):
        if cmd in [EDACMD.da_m_mine1]:
            return True
        return False


