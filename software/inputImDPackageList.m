function ImDpackList = inputImDPackageList()
    % inputImDPackageList - Returns a list of available packages using 
    % ImageData as input instead of MovieData or MovieList.
    %
    % Purpose:
    % This function is designed to handle packages that use ImageData (ImD) 
    % as input. It addresses issues related to switching between multiple 
    % ImDs and displaying their paths on the packageGUI. 
    % 
    % The function is called in the function below to handle ImD-based 
    % input packages. In the future, instead of editing each function individually, 
    % simply add new ImD input packages to inputImDPackageList as needed.
    %
    % It is used in the following function:
    %
    %   - packageGUI_RefreshFcn.m: 
    %       Solved the issue of ImageData paths not being correctly displayed 
    %       on the packageGUI.
    %
    % Author:
    %   Qiongjing (Jenny) Zou, Oct 2024
%
% Copyright (C) 2025, Danuser Lab - UTSouthwestern 
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

    % Define the list of available packages
    ImDpackList = {
        'FishATLASPackage', ...
        % Add new packages here
    };
end