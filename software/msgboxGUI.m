function varargout = msgboxGUI(varargin)
% MSGBOXGUI M-file for msgboxGUI.fig
%      MSGBOXGUI, by itself, creates a new MSGBOXGUI or raises the existing
%      singleton*.
%
%      H = MSGBOXGUI returns the handle to a new MSGBOXGUI or the handle to
%      the existing singleton*.
%
%      MSGBOXGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MSGBOXGUI.M with the given input arguments.
%
%      MSGBOXGUI('Property','Value',...) creates a new MSGBOXGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before msgboxGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to msgboxGUI_OpeningFcn via varargin.
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

% Edit the above text to modify the response to help msgboxGUI

% Last Modified by GUIDE v2.5 07-Sep-2011 11:43:10

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @msgboxGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @msgboxGUI_OutputFcn, ...
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


% --- Executes just before msgboxGUI is made visible.
function msgboxGUI_OpeningFcn(hObject, eventdata, handles, varargin)
%
% fhandle = msgboxGUI
%
% fhandle = msgboxGUI(paramName, 'paramValue', ... )
%
% Help dialog GUI
% 
% Parameter Field Names:
%       'text' -> Help text
%       'title' -> Title of text box
%       'name'-> Name of dialog box
%

% Choose default command line output for msgboxGUI
handles.output = hObject;

% Parse input
ip = inputParser;
ip.CaseSensitive=false;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addParamValue('text','',@ischar);
ip.addParamValue('extendedText','',@ischar);
ip.addParamValue('name','',@ischar);
ip.addParamValue('title','',@ischar);
ip.parse(hObject,eventdata,handles,varargin{:});

% Apply input
set(hObject, 'Name',ip.Results.name)
set(handles.edit_text, 'String',ip.Results.text)
set(handles.text_title, 'string',ip.Results.title)

if ~isempty(ip.Results.extendedText),
    userData = get(handles.figure1,'UserData');

    if isempty(userData), userData = struct(); end
    set(handles.pushbutton_extendedText,'Visible','on');
    userData.text = ip.Results.text;
    userData.extendedText=ip.Results.extendedText;
    userData.type='basic';
    set(handles.figure1,'UserData',userData);
end

% Update handles structure
uicontrol(handles.pushbutton_done)
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = msgboxGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)
delete(handles.figure1)


% --- Executes on key press with focus on figure1 or any of its controls.
function figure1_WindowKeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
if strcmp(eventdata.Key, 'return'), delete(hObject); end


% --- Executes on button press in pushbutton_extendedText.
function pushbutton_extendedText_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_extendedText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

userData=get(handles.figure1,'UserData');

if isempty(userData), userData = struct(); end

if strcmp(userData.type,'basic')
    userData.type='extended';
    set(handles.edit_text, 'String',userData.extendedText);
    set(hObject,'String','See basic report...');
else 
    userData.type='basic';
    set(handles.edit_text, 'String',userData.text);
    set(hObject,'String','See extended report...');
end
set(handles.figure1,'UserData',userData);
guidata(hObject,handles);
