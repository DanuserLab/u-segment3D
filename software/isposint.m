function isit = isposint(in)
%ISPOSINT returns true if all elements of the input are positive integers
%
% tf = isposint(in)
%
% This function duplicates the functionality of isposintscalar and
% isposintmat, but those functions are in the system identification
% toolbox, so this avoids having your code depend on that toolbox.
%
% Hunter Elliott
% 4/2011
%
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
isit = isnumeric(in) & in > 0 & abs(round(in)) == in;