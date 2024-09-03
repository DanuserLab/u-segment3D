function userfcn_drawIcon(handles, type, ID, msg, savetogui)
% This function draw or clears process status icons on package GUI
% The tag of checkboxes of processes are defined as the following format:
% 
% axes_icon_1, axes_icon_2, ..., axes_icon_n
%
% Input:
%   handles - structure handles of package GUI
%   type - the type of icons, if type = 'clear', then clears the icons of
%          processes specified by 'ID'
%   ID - array of the id of processes to draw icons specified by type and
%        msg
%   msg (optional) - icon message
%   savetogui (optional) - true or false, if save icon and message to user data,
%                          default is true
%   userData (optional) - if savetogui = true, userData is passed by main function
%
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

userData = get(handles.figure1, 'UserData');

if nargin < 4
    msg = '';
    savetogui = false;
    
elseif nargin < 5
    savetogui = false;

end

switch type
    case 'pass'
        iconData = userData.passIconData;
        
    case 'error'
        iconData = userData.errorIconData;
        
    case 'warn'
        iconData = userData.warnIconData;
        
    case 'clear'
        for i = ID
            cla(handles.(['axes_icon_' ,num2str(i)]));
            if savetogui
               % Save icon and message to user data
               userData.statusM(userData.id).IconType{i} = [];
               userData.statusM(userData.id).Msg{i} = [];
               set(handles.figure1, 'UserData', userData)                
            end
        end
        return
        
    otherwise
        error('User-defined: function input ''type'' is incorrect');
        
end

for i = ID
    
    axes(handles.(['axes_icon_',num2str(i)]));
    Img = image(iconData);
    set(gca, 'Tag', ['axes_icon_' num2str(i)], 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'), 'visible','off','YDir','reverse' )
    set(Img,'ButtonDownFcn',@icon_ButtonDownFcn)
    set(Img,'UserData',msg)
    
    if savetogui
        % Save icon and message to user data
        userData.statusM(userData.id).IconType{i} = type;
        userData.statusM(userData.id).Msg{i} = msg;
        set(handles.figure1, 'UserData', userData)    
    end

end
