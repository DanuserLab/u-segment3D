function varargout = abstractProcessGUI(varargin)
%ABSTRACTPROCESSGUI M-file for abstractProcessGUI.fig
%      ABSTRACTPROCESSGUI, by itself, creates a new ABSTRACTPROCESSGUI or raises the existing
%      singleton*.
%
%      H = ABSTRACTPROCESSGUI returns the handle to a new ABSTRACTPROCESSGUI or the handle to
%      the existing singleton*.
%
%      ABSTRACTPROCESSGUI('Property','Value',...) creates a new ABSTRACTPROCESSGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to abstractProcessGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      ABSTRACTPROCESSGUI('CALLBACK') and ABSTRACTPROCESSGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in ABSTRACTPROCESSGUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
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

% Edit the above text to modify the response to help abstractProcessGUI

% Last Modified by GUIDE v2.5 09-Jul-2012 10:49:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @abstractProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @abstractProcessGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before abstractProcessGUI is made visible.
function abstractProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)

processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:})

% Get current package and process
userData = get(handles.figure1, 'UserData');

procName = eval([userData.crtProcClassName '.getName()']); 
set(handles.uipanel_methods, 'Title', [procName ' methods']);
set(handles.text_methods, 'String', ['Choose a ' lower(procName) ' method']);


% Get current process constructer, set-up GUIs and mask refinement process
% constructor
try 
    userData.subProcClassNames = eval([userData.crtProcClassName '.getConcreteClasses(userData.MD)']);
catch
    warning('Attempted to pass MovieData object to retreive appropropriate concrete classes');
    userData.subProcClassNames = eval([userData.crtProcClassName '.getConcreteClasses()']);
end

isGraphicalProcess = @(x) Process.isProcess(x) && Process.hasGUI(x);
validClasses = cellfun(isGraphicalProcess, userData.subProcClassNames);
userData.subProcClassNames = userData.subProcClassNames(validClasses);
userData.subProcConstr = cellfun(@(x) str2func(x),userData.subProcClassNames,'Unif',0);
userData.subProcGUI = cellfun(@(x) eval([x '.GUI']),userData.subProcClassNames,'Unif',0);
subProcNames = cellfun(@(x) eval([x '.getName']),userData.subProcClassNames,'Unif',0);
popupMenuProcName = vertcat(subProcNames,{['Choose a ' lower(procName) ' method']});

% Set up input channel list box
if isempty(userData.crtProc)
    value = numel(userData.subProcClassNames)+1;
    set(handles.pushbutton_set, 'Enable', 'off');
else
%     value = find(strcmp(userData.crtProc.getName,subProcNames));
    value = find(cellfun(@isempty, strfind(subProcNames, userData.crtProc.getName))==0,1);
    if isempty(value); value = numel(subProcNames)+1; end
end

existSubProc = @(proc) any(cellfun(@(x) isa(x,proc),userData.MD.processes_));
for i=find(cellfun(existSubProc,userData.subProcClassNames'))
  popupMenuProcName{i} = ['<html><b>' popupMenuProcName{i} '</b></html>'];
end

set(handles.popupmenu_methods, 'String', popupMenuProcName,...
    'Value',value)

% Choose default command line output for abstractProcessGUI
handles.output = hObject;

% Update user data and GUI data
set(hObject, 'UserData', userData);
guidata(hObject, handles);

% % if only one selection available, just select it!
% if numel(subProcNames) == 1
%     set(handles.popupmenu_methods, 'Value', 1);
%     abstractProcessGUI('pushbutton_set_Callback', handles.pushbutton_set,[], handles);
% 
%     % set select to 1
%     % click button
% %     
% %     segProcID = 1;
% %     subProcGUI = userData.subProcGUI{segProcID};
% %     subProcGUI('mainFig',userData.mainFig,userData.procID,...
% %         'procConstr',userData.subProcConstr{segProcID},...
% %         'procClassName',userData.subProcClassNames{segProcID});
% %     delete(handles.figure1);
% end




% --- Outputs from this function are returned to the command line.
function varargout = abstractProcessGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
% if ~isempty(handles)
    varargout{1} = handles.output;
% else
%     varargout{1} = [];
% end


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)
% Delete figure
delete(handles.figure1);


% --- Executes on selection change in popupmenu_methods.
function popupmenu_methods_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_methods (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_methods contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_methods
content = get(hObject, 'string');
if get(hObject, 'Value') == length(content)
    set(handles.pushbutton_set, 'Enable', 'off')
else
    set(handles.pushbutton_set, 'Enable', 'on')
end

% --- Executes on button press in pushbutton_set.
function pushbutton_set_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');
segProcID = get(handles.popupmenu_methods, 'Value');
subProcGUI = userData.subProcGUI{segProcID};
subProcGUI('mainFig',userData.mainFig,userData.procID,...
    'procConstr',userData.subProcConstr{segProcID},...
    'procClassName',userData.subProcClassNames{segProcID});
delete(handles.figure1);


% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');
if ~isempty(userData) %%%% check and see why
delete(userData.helpFig(ishandle(userData.helpFig))); 
end
set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);
