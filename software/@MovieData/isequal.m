function [ tf ] = isequal( obj, MD, varargin )
%isequal Compare two or more MovieData objects
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
     if(isequal(size(obj),size(MD)))
         if(numel(obj) > 1)
             tf = all(arrayfun(@isequal,obj,MD));
         elseif(isempty(obj))
             tf = true;
         elseif(isa(obj,'MovieData') && isa(MD,'MovieData'))
             % Two MovieData are the same if they will be saved in the same
             % place
             tf = obj == MD || ...
                  strcmp(obj.movieDataPath_,MD.movieDataPath_) && ...
                  strcmp(obj.movieDataFileName_,MD.movieDataFileName_);
         else
             tf = false;
         end
     else
         tf = false;
     end
     if(tf && nargin > 2)
         tf = isequal(obj,varargin{1},varargin{2:end});
     end
end

