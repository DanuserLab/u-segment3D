function varargout = recycleProcessGUI(varargin)
% RECYCLEPROCESSGUI M-file for recycleProcessGUI.fig
%      RECYCLEPROCESSGUI, by itself, creates a new RECYCLEPROCESSGUI or raises the existing
%      singleton*.
%
%      H = RECYCLEPROCESSGUI returns the handle to a new RECYCLEPROCESSGUI or the handle to
%      the existing singleton*.
%
%      RECYCLEPROCESSGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RECYCLEPROCESSGUI.M with the given input arguments.
%
%      RECYCLEPROCESSGUI('Property','Value',...) creates a new RECYCLEPROCESSGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before recycleProcessGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to recycleProcessGUI_OpeningFcn via varargin.
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

% Edit the above text to modify the response to help recycleProcessGUI

% Last Modified by GUIDE v2.5 27-Sep-2011 11:06:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @recycleProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @recycleProcessGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
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


% --- Executes just before recycleProcessGUI is made visible.
function recycleProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% 
% recycleProcessGUI(process, package, 'mainFig', handles.figure1)
% 
% Input:
%
%   process - the array of processes for recycle
%   package - the package where the processes would be attached to
%
% User Data:
% 
% userData.recyclableProc - the array of processes for recycle
% userData.package - the package where the processes would be attached to
% 
% userData.mainFig - handle of movie selector GUI
% 
% 
% Input check
ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addRequired('process',@(x) all(cellfun(@(y) isa(y,'Process'),x)));
ip.addRequired('package',@(x) isa(x,'Package'));
ip.addParamValue('mainFig',-1,@ishandle);
ip.parse(hObject,eventdata,handles,varargin{:})

% Store input
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
userData.recyclableProc =ip.Results.process  ; 
userData.package = ip.Results.package;
userData.mainFig=ip.Results.mainFig;
userData.previewFig = -1;

set(handles.text_copyright, 'String', getLCCBCopyright())

% Choose default command line output for recycleProcessGUI
handles.output = hObject;
    
% GUI set-up
set(handles.text_package, 'String', userData.package.getName)
set(handles.text_movie, 'String', [userData.package.owner_.getPath filesep userData.package.owner_.getFilename])

% Create recyclable processes list
nProc = numel(userData.recyclableProc);
string = cell(1,nProc);
for i = 1:nProc
    string{i} = userData.recyclableProc{i}.getName;
end
set(handles.listbox_processes, 'String',string, 'UserData',1:nProc)

% Create package processes list
nProc = numel(userData.package.processes_);
string = cell(1,nProc);
for i = 1:nProc
    processClassName = userData.package.getProcessClassNames{i};
    processName=eval([processClassName '.getName']);
    string{i} = [' Step ' num2str(i) ': ' processName];
end
set(handles.listbox_package, 'String', string,'UserData',cell(1,nProc));

set(handles.figure1,'UserData',userData)

uiwait(handles.figure1)
guidata(hObject, handles);

% UIWAIT makes recycleProcessGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = recycleProcessGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

delete(handles.figure1)


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)

uiresume(handles.figure1)


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)

% Add the processes to the new package
userData = get(handles.figure1, 'UserData');

procIds = get(handles.listbox_package, 'UserData');
validPackageProc=~cellfun(@isempty,procIds);

if isempty(find(validPackageProc,1)), uiresume(handles.figure1); end

for i = find(validPackageProc)
    userData.package.setProcess(i,userData.recyclableProc{procIds{i}});
end

uiresume(handles.figure1)

% --- Executes on button press in pushbutton_add.
function pushbutton_add_Callback(hObject, eventdata, handles)

% Test if ther is any process to dispatch
processString = get(handles.listbox_processes, 'String');
if isempty(processString), return; end

% Load process list and get corresponding process id
userData=get(handles.figure1,'UserData');
props = get(handles.listbox_processes, {'UserData','Value'});
procIds=props{1};
procIndex = props{2};
procId=procIds(procIndex);
process= userData.recyclableProc(procId);

% Find associated  package process
getPackId = @(proc)find(cellfun(@(x) isa(proc,x),userData.package.getProcessClassNames));
packProcIds = cellfun(getPackId,process);
if any(packProcIds) 
   % Update package process and pids
   props = get(handles.listbox_package, {'String','UserData'});
   packageString = props{1};
   for i = 1:numel(packProcIds)
       props{2}{packProcIds(i)}=procId(i);
       packageString{packProcIds(i)} = ['<html><b> ' packageString{packProcIds(i)} '</b></html>'];
   end
   set(handles.listbox_package, 'String',packageString','UserData',props{2});
   
   % Remove process from the list
   processString(procIndex) = [];
   procIds(procIndex) = [];
   % Set the highlighted value
   if ~isscalar(procIndex) || (procIndex > length(processString) && procIndex > 1)
       set(handles.listbox_processes, 'Value', length(processString));
   end
   set(handles.listbox_processes, 'String', processString, 'UserData', procIds)
end

% --- Executes on button press in pushbutton_remove.
function pushbutton_remove_Callback(hObject, eventdata, handles)

% Find package process id and return if empty
props = get(handles.listbox_package,{'UserData','Value','String'});
procIds =props{1};
procId = procIds{props{2}};
if isempty(procId), return; end

% Remove pid from the package id list and update package string
userData=get(handles.figure1,'UserData');
packageString = props{3};
procIds(props{2})=[];
processClassName = userData.package.getProcessClassNames{props{2}};
processName=eval([processClassName '.getName']);
packageString{props{2}} = [' Step ' num2str(props{2}) ': ' processName];
set(handles.listbox_package,'String',packageString,'UserData',procIds);

% Add process to the process list
props = get(handles.listbox_processes, {'UserData','String'});
procIds=props{1};
processString = props{2};
processString{end+1} = userData.recyclableProc{procId}.getName;
set(handles.listbox_processes,'String',processString,'UserData',[procIds procId]);

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isequal(get(handles.figure1, 'waitstatus'), 'waiting')
    % The GUI is still in UIWAIT, us UIRESUME
    uiresume(handles.figure1);
else
    % The GUI is no longer waiting, just close it
    delete(handles.figure1);
end

% --- Executes on button press in pushbutton_preview.
function pushbutton_preview_Callback(hObject, eventdata, handles)

% Test if ther is any process to preview
processString = get(handles.listbox_processes, 'String');
if isempty(processString), return; end

% Load process list and get corresponding process id
userData=get(handles.figure1,'UserData');
props = get(handles.listbox_processes, {'UserData','Value'});
procIds=props{1};
procIndex = props{2};
procId=procIds(procIndex);

userData.previewFig = userData.recyclableProc{procId}.resultDisplay();

% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');

if ishandle(userData.previewFig), delete(userData.detailFig);  end
