function varargout = ImagePreprocessingProcessGUI(varargin)
%IMAGEPREPROCESSINGPROCESSGUI MATLAB code file for ImagePreprocessingProcessGUI.fig
%      IMAGEPREPROCESSINGPROCESSGUI, by itself, creates a new IMAGEPREPROCESSINGPROCESSGUI or raises the existing
%      singleton*.
%
%      H = IMAGEPREPROCESSINGPROCESSGUI returns the handle to a new IMAGEPREPROCESSINGPROCESSGUI or the handle to
%      the existing singleton*.
%
%      IMAGEPREPROCESSINGPROCESSGUI('Property','Value',...) creates a new IMAGEPREPROCESSINGPROCESSGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to ImagePreprocessingProcessGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      IMAGEPREPROCESSINGPROCESSGUI('CALLBACK') and IMAGEPREPROCESSINGPROCESSGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in IMAGEPREPROCESSINGPROCESSGUI.M with the given input
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

% Edit the above text to modify the response to help ImagePreprocessingProcessGUI

% Last Modified by GUIDE v2.5 16-Jul-2024 13:21:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImagePreprocessingProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @ImagePreprocessingProcessGUI_OutputFcn, ...
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


% --- Executes just before ImagePreprocessingProcessGUI is made visible.
function ImagePreprocessingProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)

processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',1);

% Parameter setup
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;


% set GUI with Parameters

set(handles.edit_factor, 'String',num2str(funParams.factor))
set(handles.edit_voxel_res, 'String',num2str(funParams.voxel_res))

% set below two edit boxes anyway, even they depend on checkbox_do_bg_correction. b/c checkbox_do_bg_correction_Callback does not set them
set(handles.edit_bg_ds, 'String',num2str(funParams.bg_ds))
set(handles.edit_bg_sigma, 'String',num2str(funParams.bg_sigma))
if funParams.do_bg_correction
    set(handles.checkbox_do_bg_correction, 'Value', 1);
else
    set(handles.checkbox_do_bg_correction, 'Value', 0);
    set(get(handles.uipanel_uneven,'Children'),'Enable','off');
end

set(handles.edit_normalize_min, 'String',num2str(funParams.normalize_min))
set(handles.edit_normalize_max, 'String',num2str(funParams.normalize_max))

% Update user data and GUI data
handles.output = hObject;
set(handles.figure1, 'UserData', userData);
guidata(hObject, handles);



% --- Outputs from this function are returned to the command line.
function varargout = ImagePreprocessingProcessGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end

if isfield(userData, 'helpFig') && ishandle(userData.helpFig)
   delete(userData.helpFig) 
end

set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);


% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
% Key: name of the key that was pressed, in lower case
% Character: character interpretation of the key(s) that was pressed
% Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1);


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(hObject, eventdata, handles)

%  Check user input --------
if isempty(get(handles.listbox_selectedChannels, 'String'))
   errordlg('Please select at least one input channel from ''Available Channels''.','Setting Error','modal') 
    return;
end


if isnan(str2double(get(handles.edit_factor, 'String'))) ...
        || str2double(get(handles.edit_factor, 'String')) <= 0
    errordlg('Please provide a valid input for ''Isotropic Scaling Factor''.','Setting Error','modal');
    return;
end

inputStr = get(handles.edit_voxel_res, 'String');
if isempty(inputStr) ...
    || any(isnan(sscanf(inputStr, '%d')')) ...
    || any(sscanf(inputStr, '%d')' < 0) ...
    || length(sscanf(inputStr, '%d')') ~= length(strsplit(inputStr))
  errordlg('Please provide a valid input for ''Pixel Size (to correct for anisotropic imaging)''.','Setting Error','modal');
  return;
end

if get(handles.checkbox_do_bg_correction, 'value')
    if isnan(str2double(get(handles.edit_bg_ds, 'String'))) ...
            || str2double(get(handles.edit_bg_ds, 'String')) <= 0
        errordlg('Please provide a valid input for ''Downsample Scale Factor''.','Setting Error','modal');
        return;
    end 
    if isnan(str2double(get(handles.edit_bg_sigma, 'String'))) ...
            || str2double(get(handles.edit_bg_sigma, 'String')) <= 0
        errordlg('Please provide a valid input for ''Smooth Sigma''.','Setting Error','modal');
        return;
    end   
end

if isnan(str2double(get(handles.edit_normalize_min, 'String'))) ...
        || str2double(get(handles.edit_normalize_min, 'String')) < 0 ...
        || str2double(get(handles.edit_normalize_min, 'String')) > 100
    errordlg('Please provide a valid input for ''Lower Intensity Cutoff''.','Setting Error','modal');
    return;
end

if isnan(str2double(get(handles.edit_normalize_max, 'String'))) ...
        || str2double(get(handles.edit_normalize_max, 'String')) < 0 ...
        || str2double(get(handles.edit_normalize_max, 'String')) > 100
    errordlg('Please provide a valid input for ''Upper Intensity Cutoff''.','Setting Error','modal');
    return;
end


%  Process Sanity check ( only check underlying data )
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
try
    userData.crtProc.sanityCheck;
catch ME
    errordlg([ME.message 'Please double check your data.'],...
                'Setting Error','modal');
    return;
end

% Retrieve GUI-defined parameters
channelIndex = get(handles.listbox_selectedChannels, 'Userdata');
funParams.ChannelIndex = channelIndex;

funParams.factor = str2double(get(handles.edit_factor, 'String'));
funParams.voxel_res = sscanf(get(handles.edit_voxel_res, 'String'), '%d')';

if get(handles.checkbox_do_bg_correction, 'Value')
    funParams.do_bg_correction = true;
    funParams.bg_ds = str2double(get(handles.edit_bg_ds, 'String'));
    funParams.bg_sigma = str2double(get(handles.edit_bg_sigma, 'String'));
else
    funParams.do_bg_correction = false;
end

funParams.normalize_min = str2double(get(handles.edit_normalize_min, 'String'));
funParams.normalize_max = str2double(get(handles.edit_normalize_max, 'String'));


% Set parameters and update main window
processGUI_ApplyFcn(hObject, eventdata, handles,funParams);



% --- Executes on key press with focus on pushbutton_done and none of its controls.
function pushbutton_done_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_done (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_done, [], handles);
end



% --- Executes on button press in checkbox_do_bg_correction.
function checkbox_do_bg_correction_Callback(hObject, eventdata, handles)
% Hint: get(hObject,'Value') returns toggle state of checkbox_do_bg_correction_Callback

if get(hObject, 'Value')
    set(get(handles.uipanel_uneven,'Children'),'Enable','on');
else
    set(get(handles.uipanel_uneven,'Children'),'Enable','off');
end
