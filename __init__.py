# __all__ = ["metagen", "readdata", "S11fit", "newfile"]

# Loads utils with shorter names
# File creation, callable as stanalysis.newfile(...)
from stanalysis.utils.newfile import newfile

# Metagen creation, callable as stanalysis.metagen.fromlimits(...)
import stanalysis.utils.metagen as metagen

# File reading, callable as stanalysis.readdata.readQUCS(...)
import stanalysis.utils.readdata as readdata

# File writing, callable as stanalysis.savetxt(...)
from stanalysis.utils.writematrix import writematrix as savetxt
from stanalysis.utils.writematrix import writedict as savedict
from stanalysis.utils.writematrix import writedictarray as savedictarray
from stanalysis.utils.writematrix import writeparams as writeparams
from stanalysis.utils.writematrix import writeparnames as writeparnames
from stanalysis.utils.writematrix import params_to_str as params_to_str
from stanalysis.utils.writematrix import writeline as writeline
from stanalysis.utils.writematrix import writeframe as saveframe
from stanalysis.utils.writematrix import writeframearray as saveframearray
from stanalysis.utils.stlabdict import stlabdict
from stanalysis.utils.stlabdict import stlabmtx
from stanalysis.utils.stlabdict import framearr_to_mtx

# Fitting routines, callable as stanalysis.S11fit(...)
from stanalysis.utils.S11fit import fit as S11fit
from stanalysis.utils.S11fit import S11full as S11func
from stanalysis.utils.S11fit import backmodel as S11back
from stanalysis.utils.S11fit import S11theo as S11theo

# Autoplotter, callable as stanalysis.autoplot(...)
from stanalysis.utils.autoplotter import autoplot
