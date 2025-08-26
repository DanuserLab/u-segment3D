function MLpackList = inputMLPackageList()
    % inputMLPackageList - Returns a list of available packages using 
    % MovieList as input instead of MovieData.
    %
    % Purpose:
    % This function is designed to handle packages that use MovieList (ML) 
    % as input. It addresses issues related to path display, switching 
    % between MovieLists, and managing multiple MovieLists on the packageGUI 
    % and setting GUIs. 
    % The function is called in five different functions below
    % to handle ML-based input packages, so in the future, instead of 
    % editing each of these five functions, you can simply add new ML input 
    % packages to inputMLPackageList as needed.
    %
    % It is used in below functions:
    %   - packageGUI.m: 
    %       (1) Fixed the error when switching between MovieLists on packageGUI.
    %       (2) Enabled the Folder Icon (pushbutton_open) to call another GUI, 
    %           named folderViewer, for opening multiple result folders.
    %
    %   - packageGUI_RefreshFcn.m: 
    %       Solved the issue of MovieLists' paths not being correctly displayed 
    %       on the packageGUI.
    %
    %   - packageGUI_RunFcn.m: 
    %       Fixed the issue packageGUI cannot handle multiple MLs
    %
    %   - processGUI_OpeningFcn.m: 
    %       Fixed the issue where, if only one MovieList is present, the 
    %       checkbox (Apply Check/Uncheck to All Movies) would remain unchecked 
    %       and invisible on PackageGUI and on all setting GUIs for the ML
    %       as input packages.
    %
    %   - userfcn_checkAllMovies.m: 
    %       Resolved the issue where the checkbox (Apply Check/Uncheck to All 
    %       Movies) was not functioning properly on packageGUI.
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
    MLpackList = {
        'XcorrFluctuationPackage', ...
        'GrangerCausalityAnalysisPackage', ...
        % Add new packages here
    };
end