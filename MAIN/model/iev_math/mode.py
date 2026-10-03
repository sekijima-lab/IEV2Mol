import os
def old_compatible():
    value=os.environ.get('IEV2MOL_OLD_COMPATIBLE','1')
    if value not in ('0','1'):raise ValueError('IEV2MOL_OLD_COMPATIBLE must be 0 or 1')
    return value=='1'
