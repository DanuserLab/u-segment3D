function varargout = CellposeSegmentationProcessGUI(varargin)
%CELLPOSESEGMENTATIONPROCESSGUI MATLAB code file for CellposeSegmentationProcessGUI.fig
%      CELLPOSESEGMENTATIONPROCESSGUI, by itself, creates a new CELLPOSESEGMENTATIONPROCESSGUI or raises the existing
%      singleton*.
%
%      H = CELLPOSESEGMENTATIONPROCESSGUI returns the handle to a new CELLPOSESEGMENTATIONPROCESSGUI or the handle to
%      the existing singleton*.
%
%      CELLPOSESEGMENTATIONPROCESSGUI('Property','Value',...) creates a new CELLPOSESEGMENTATIONPROCESSGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to CellposeSegmentationProcessGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      CELLPOSESEGMENTATIONPROCESSGUI('CALLBACK') and CELLPOSESEGMENTATIONPROCESSGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in CELLPOSESEGMENTATIONPROCESSGUI.M with the given input
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

% Edit the above text to modify the response to help CellposeSegmentationProcessGUI

% Last Modified by GUIDE v2.5 17-Jul-2024 11:58:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CellposeSegmentationProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @CellposeSegmentationProcessGUI_OutputFcn, ...
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


% --- Executes just before CellposeSegmentationProcessGUI is made visible.
function CellposeSegmentationProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)
processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',1);

% Parameter setup
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;


% set GUI with Parameters

% set below two edit boxes anyway, even they depend on checkbox_hist_norm. b/c checkbox_hist_norm_Callback does not set them
set(handles.edit_histnorm_kernel_size, 'String', num2str([funParams.histnorm_kernel_size{:}])) % this is how to convert funParams.histnorm_kernel_size = {int16(64), int16(64)} to string.
set(handles.edit_histnorm_clip_limit, 'String',num2str(funParams.histnorm_clip_limit))
if funParams.hist_norm
    set(handles.checkbox_hist_norm, 'Value', 1);
else
    set(handles.checkbox_hist_norm, 'Value', 0);
    set(get(handles.uipanel_histnorm,'Children'),'Enable','off');
end

%Setup Cellpose Model pop up menu:
set(handles.popupmenu_cellpose_modelname, 'String', CellposeSegmentationProcess.getValidCellposeModelname);
parVal = funParams.cellpose_modelname;
valSel  = find(ismember(CellposeSegmentationProcess.getValidCellposeModelname, parVal));
if isempty(valSel), valSel = 1; end
set(handles.popupmenu_cellpose_modelname, 'Value', valSel);

%Setup Cellpose Color pop up menu:
set(handles.popupmenu_cellpose_channels, 'String', CellposeSegmentationProcess.getValidCellposeChannels);
parVal = funParams.cellpose_channels;
valSel  = find(ismember(CellposeSegmentationProcess.getValidCellposeChannels, parVal));
if isempty(valSel), valSel = 1; end
set(handles.popupmenu_cellpose_channels, 'Value', valSel);


if funParams.use_Cellpose_auto_diameter
    set(handles.checkbox_use_Cellpose_auto_diameter, 'Value', 1)
else
    set(handles.checkbox_use_Cellpose_auto_diameter, 'Value', 0)
end

if funParams.model_invert
    set(handles.checkbox_model_invert, 'Value', 1)
else
    set(handles.checkbox_model_invert, 'Value', 0)
end


set(handles.edit_ksize, 'String',num2str(funParams.ksize))
set(handles.edit_best_diam, 'String',num2str(funParams.best_diam))
set(handles.edit_diam_range, 'String', funParams.diam_range) % funParams.diam_range = 'np.arange(10,121,2.5)' which is a string
set(handles.edit_smoothwinsize, 'String',num2str(funParams.smoothwinsize))
set(handles.edit_test_slice, 'String',num2str(funParams.test_slice))


%Setup "Use Edge Magnitude to Set Best Slice" pop up menu, and it depends on edit_test_slice
set(handles.popupmenu_use_edge, 'String', CellposeSegmentationProcess.getValidUseEdge);
if funParams.use_edge % is a logical, true is dropdown 1, false is dropdown 2
    parVal = 'Use edge strength to determine optimal slice';
else
    parVal = 'Use maximum intensity to determine optimal slice';
end
valSel  = find(ismember(CellposeSegmentationProcess.getValidUseEdge, parVal));
if isempty(valSel), valSel = 1; end
set(handles.popupmenu_use_edge, 'Value', valSel);
if ~ismissing(funParams.test_slice)
    set(handles.text_use_edge, 'Enable','off');
    set(handles.popupmenu_use_edge, 'Enable','off');
end


% Update user data and GUI data
handles.output = hObject;
set(handles.figure1, 'UserData', userData);
guidata(hObject, handles);



% --- Outputs from this function are returned to the command line.
function varargout = CellposeSegmentationProcessGUI_OutputFcn(hObject, eventdata, handles)
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


if get(handles.checkbox_hist_norm, 'value')
    % histnorm_kernel_size needs to be 2 positive integers
    inputStr = get(handles.edit_histnorm_kernel_size, 'String');
    if isempty(inputStr) ...
        || any(isnan(sscanf(inputStr, '%f')')) ...
        || any(sscanf(inputStr, '%f')' <= 0) ...
        || any(floor(sscanf(inputStr, '%f')') ~= sscanf(inputStr, '%f')') ...
        || length(sscanf(inputStr, '%f')') ~= 2 ...
        || length(sscanf(inputStr, '%f')') ~= length(strsplit(inputStr))
      errordlg('Please provide a valid input for ''Kernel Size of 2D Histogram Equalization''.','Setting Error','modal');
      return;
    end

    if isnan(str2double(get(handles.edit_histnorm_clip_limit, 'String'))) ...
            || str2double(get(handles.edit_histnorm_clip_limit, 'String')) < 0
        errordlg('Please provide a valid input for ''Clip Limit in 2D Histogram Equalization''.','Setting Error','modal');
        return;
    end   
end


if isnan(str2double(get(handles.edit_ksize, 'String'))) ...
        || str2double(get(handles.edit_ksize, 'String')) <= 1 ...
        || floor(str2double(get(handles.edit_ksize, 'String'))) ~= str2double(get(handles.edit_ksize, 'String'))
    errordlg('Please provide a valid input for ''Contrast Score Neighborhood Size''.','Setting Error','modal');
    return;
end

% best_diam can be empty (None, string(missing) in Matlab) or int >=1
if ~isempty(get(handles.edit_best_diam, 'String')) && (str2double(get(handles.edit_best_diam, 'String')) < 1 ...
        || floor(str2double(get(handles.edit_best_diam, 'String'))) ~= str2double(get(handles.edit_best_diam, 'String')))
    errordlg('Please provide a valid input for ''Cellpose Diameter''.','Setting Error','modal');
    return;
end

% did not do user input check for diam_range ('np.arange(10,121,2.5)') here. - QZ TODO

if isnan(str2double(get(handles.edit_smoothwinsize, 'String'))) ...
        || str2double(get(handles.edit_smoothwinsize, 'String')) < 3 ...
        || floor(str2double(get(handles.edit_smoothwinsize, 'String'))) ~= str2double(get(handles.edit_smoothwinsize, 'String'))
    errordlg('Please provide a valid input for ''Smooth Contrast Function Window Size''.','Setting Error','modal');
    return;
end

% test_slice can be empty (None, string(missing) in Matlab) or int >=0
if ~isempty(get(handles.edit_test_slice, 'String')) && (str2double(get(handles.edit_test_slice, 'String')) < 0 ...
        || floor(str2double(get(handles.edit_test_slice, 'String'))) ~= str2double(get(handles.edit_test_slice, 'String')))
    errordlg('Please provide a valid input for ''Representative 2D Slice''.','Setting Error','modal');
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

if get(handles.checkbox_hist_norm, 'Value')
    funParams.hist_norm = true;
    funParams.histnorm_kernel_size = num2cell(sscanf(get(handles.edit_histnorm_kernel_size, 'String'), '%d')'); % need to be 2 integer ('%d') in 1x2 cell array
    funParams.histnorm_clip_limit = str2double(get(handles.edit_histnorm_clip_limit, 'String'));
else
    funParams.hist_norm = false;
end

selType = get(handles.popupmenu_cellpose_modelname, 'Value'); 
funParams.cellpose_modelname = CellposeSegmentationProcess.getValidCellposeModelname{selType};

selType = get(handles.popupmenu_cellpose_channels, 'Value'); 
funParams.cellpose_channels = CellposeSegmentationProcess.getValidCellposeChannels{selType};

if get(handles.checkbox_use_Cellpose_auto_diameter, 'Value')
    funParams.use_Cellpose_auto_diameter = true;
else
    funParams.use_Cellpose_auto_diameter = false;
end

if get(handles.checkbox_model_invert, 'Value')
    funParams.model_invert = true;
else
    funParams.model_invert = false;
end

funParams.ksize = int16(str2double(get(handles.edit_ksize, 'String'))); % this parameter need to be a integer!

% best_diam can be empty (None, string(missing) in Matlab) or int >=1
if isempty(get(handles.edit_best_diam, 'String'))
    funParams.best_diam = string(missing);
else
    funParams.best_diam = int16(str2double(get(handles.edit_best_diam, 'String')));
end

funParams.diam_range = get(handles.edit_diam_range, 'String');

funParams.smoothwinsize = int16(str2double(get(handles.edit_smoothwinsize, 'String'))); % this parameter need to be a integer!

% test_slice can be empty (None, string(missing) in Matlab) or int >=0
if isempty(get(handles.edit_test_slice, 'String'))
    funParams.test_slice = string(missing);
    % use_edge if test_slice is not set:
    if get(handles.popupmenu_use_edge, 'Value') == 1
        funParams.use_edge = true;
    elseif get(handles.popupmenu_use_edge, 'Value') == 2
        funParams.use_edge = false;
    end
else
    funParams.test_slice = int16(str2double(get(handles.edit_test_slice, 'String')));
end


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



% --- Executes on button press in checkbox_hist_norm.
function checkbox_hist_norm_Callback(hObject, eventdata, handles)
% Hint: get(hObject,'Value') returns toggle state of checkbox_hist_norm

if get(hObject, 'Value')
    set(get(handles.uipanel_histnorm,'Children'),'Enable','on');
else
    set(get(handles.uipanel_histnorm,'Children'),'Enable','off');
end

% --- Executes on selection change in edit_test_slice_Callback.
function edit_test_slice_Callback(hObject, eventdata, handles)

if isempty(get(handles.edit_test_slice, 'String'))
    % use_edge if test_slice is not set:
    set(handles.text_use_edge, 'Enable','on');
    set(handles.popupmenu_use_edge, 'Enable','on');
else
    set(handles.text_use_edge, 'Enable','off');
    set(handles.popupmenu_use_edge, 'Enable','off');
end

