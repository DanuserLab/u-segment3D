function s = loadLCCBIcons(s)
% Load set of common icons for graphical interfaces
% 
% Synopsis: icons = loadLCCBIcons(s)
% 
% Input:
%         s - an existing structure to which fields corresponding to each
%         icon will be appending. A new structure will be created if empty.
%
% Output:
%         s - a structure with a series of fields named xxxIconData which
%         value corresponds to the TrueColor icon data
%
% Sebastien Besson, June 2012
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

% Get the main path to the icons folder
iconsPath = fullfile(fileparts(which('packageGUI.m')),'icons');

% Create 
iconsNames ={'pass','accepted 48 32x32.png';'error','cancel 48 32x32.png';...
    'warn','warning 48 32x32.png';'quest','dialog-question-48x48.png';...
    'smallquest','dialog-question-22x22.png';'open','document-open-8.png'};

% Create options to set the background
bgOptions = {'BackgroundColor',get(0,'DefaultUiControlBackgroundColor')};

for i=1:size(iconsNames,1)
    s.([iconsNames{i,1} 'IconData']) = ...
        imread(fullfile(iconsPath,iconsNames{i,2}),bgOptions{:});
end

end