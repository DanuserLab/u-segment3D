function varargout = uSegment3DPackageGUI(varargin)
% Launch the GUI for the uSegment3D Package
%
% This function calls the generic packageGUI function, passes all its input
% arguments and returns all output arguments of packageGUI
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

% Qiongjing (Jenny) Zou, June 2024

if nargin>0 && isa(varargin{1},'MovieList')
    varargout{1} = packageGUI('uSegment3DPackage',...
        [varargin{1}.getMovies{:}],'ML',varargin{1},varargin{2:end});
else
    varargout{1} = packageGUI('uSegment3DPackage',varargin{:});
end
end