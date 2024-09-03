function processGUI_ApplyFcn(hObject, eventdata, handles,funParams,varargin)
%processGUI_ApplyFcn is a callback called when setting concrete process GUIs
%
%
% Sebastien Besson May 2011 (last modified Oct 2011)
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

% Add ImageData compatibility
% Updated by Qiongjing (Jenny) Zou, Jun 2020

% Check input
ip = inputParser;
ip.addRequired('hObject',@(x) ishandle(x) || isa(x, 'appdesigner.appmigration.UIControlPropertiesConverter'));
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x) || isa(x, 'event.EventData'));
ip.addRequired('handles',@isstruct);
ip.addRequired('funParams',@(x) isstruct(x) || isempty(x))
ip.addOptional('settingFcn',{},@iscell);
ip.parse(hObject,eventdata,handles,funParams,varargin{:});
settingFcn=ip.Results.settingFcn;

% if get(handles.checkbox_applytoall, 'Value')
%     confirmApplytoAll = questdlg(...
%         ['You are about to copy the current process settings to all movies.'...
%         ' Previous settings will be lost. Do you want to continue?'],...
%         'Apply settings to all movies','Yes','No','Yes');
%     if ~strcmp(confirmApplytoAll,'Yes'),
%         set(handles.checkbox_applytoall,'Value',0);
%         return
%     end
% end

% Get the main figure userData
userData = get(handles.figure1, 'UserData');
if isfield(userData,'MD'),
    field = 'MD';
elseif isfield(userData,'ML');
    field = 'ML';
elseif isfield(userData,'ImD');
    field = 'ImD';
else
    error('Missing movie');
end

% Check if the current process is equal to the package process (to cover
% empty processes as well as new subclass processes)
oldProcess = userData.crtPackage.getProcess(userData.procID);
if ~isequal(oldProcess, userData.crtProc)
    if isempty(oldProcess)
        % Create a new process and associate it to the package
        userData.(field).addProcess(userData.crtProc);
        userData.crtPackage.setProcess(userData.procID,userData.crtProc);
    else
        % Replace the process
        userData.(field).replaceProcess(oldProcess, userData.crtProc);
    end
    
end

% Override the parameters with the GUI set-up ones
parseProcessParams(userData.crtProc,funParams);

userData_main = get(userData.mainFig, 'UserData');
applytoall = isfield(handles, 'checkbox_applytoall') &&...
    get(handles.checkbox_applytoall, 'Value');
if applytoall
    moviesId = setdiff(1:numel(userData_main.(field)),userData_main.id);
else
    moviesId=[];
end

% Apply setting to all movies
for i = moviesId
    
    % Check process can
    oldProcess = userData_main.package(i).getProcess(userData.procID);
    if ~strcmp(class(oldProcess), class(userData.crtProc))
        % Create a new process for the ith move
        newProcess = userData.procConstr(userData_main.(field)(i), ...
            userData_main.package(i).outputDirectory_);
        
        if isempty(oldProcess)
            % Create a new process and associate it to the package
            userData_main.(field)(i).addProcess(newProcess);
            userData_main.package(i).setProcess(userData.procID, newProcess);
        else
            % Replace the process of the ith movie
            userData_main.(field)(i).replaceProcess(oldProcess, newProcess);
        end
    end
    
    % Override the parameters with the GUI defeined
    parseProcessParams(userData_main.package(i).getProcess(userData.procID),...
        funParams);
    
    for j=1:numel(settingFcn)
        settingFcn{j}(userData_main.package(i).getProcess(userData.procID));
    end
end

% Store the applytoall choice for this particular process
if isfield(handles, 'checkbox_applytoall')
    userData_main.applytoall(userData.procID) = ...
        get(handles.checkbox_applytoall,'Value');
end

% Aumoatically check process if settings have been set up
userData_main.statusM(userData_main.id).Checked(userData.procID) = 1;
set(userData.mainFig, 'UserData', userData_main)
if applytoall,
    userfcn_checkAllMovies(userData.procID, 1, guidata(userData.mainFig));
end

% Save user data

set(handles.figure1, 'UserData', userData);

% Refresh main screen
packageGUI_RefreshFcn(userData.handles_main,'refresh')
guidata(hObject,handles);
delete(handles.figure1);
end
