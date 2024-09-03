function userfcn_enable (index, onoff, handles, check)
% GUI tool function: this function is used to change the 'visible' property 
% of uicontrols on control panel. The name of the uicontrols are pre-defined
% in the following way: 
%       checkbox: 
%               checkbox_1  checkbox_2 ...
%
% Input: 
%       index - vector of check box index
%       onoff - enable or disable, 'on' or 'off'
%       handles - handles of control panel
%       check - (Optional) true or false. It provides a option to select/unselect 
%       the checkboxs that have been enabled/disabled.
% 
% Chuangang Ren
% 08/2010
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

if nargin < 4
    check = false;
end

for i = index(:)'
    set(handles.(['checkbox_',num2str(i)]),'Enable',onoff);
    set(handles.(['pushbutton_set_',num2str(i)]),'Enable',onoff);                                    
end

if check
    switch onoff
        case 'on'
            value=1;
        case 'off'
            value=0;
    end
    for i = 1: length(index)
        set(handles.(['checkbox_',num2str(index(i))]),'Value',value);
    end

end