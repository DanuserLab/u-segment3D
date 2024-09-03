function logic= isvalid(aDouble)
% Dirty trick to handle isvalid absence in <2013b
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 
global debuggingMode__;
if(debuggingMode__)
    warning(['isvalid:' class(aDouble) ':undefined'], ...
        ['isvalid is not defined for ' class(aDouble) ...
        '. Using always true function at ' mfilename('fullpath') '.']);
end
logic=true;
