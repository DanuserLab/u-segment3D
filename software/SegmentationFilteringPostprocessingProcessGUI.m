function varargout = SegmentationFilteringPostprocessingProcessGUI(varargin)
%SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI MATLAB code file for SegmentationFilteringPostprocessingProcessGUI.fig
%      SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI, by itself, creates a new SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI or raises the existing
%      singleton*.
%
%      H = SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI returns the handle to a new SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI or the handle to
%      the existing singleton*.
%
%      SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI('Property','Value',...) creates a new SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to SegmentationFilteringPostprocessingProcessGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI('CALLBACK') and SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in SEGMENTATIONFILTERINGPOSTPROCESSINGPROCESSGUI.M with the given input
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

% Edit the above text to modify the response to help SegmentationFilteringPostprocessingProcessGUI

% Last Modified by GUIDE v2.5 20-Aug-2024 10:38:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SegmentationFilteringPostprocessingProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @SegmentationFilteringPostprocessingProcessGUI_OutputFcn, ...
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


% --- Executes just before SegmentationFilteringPostprocessingProcessGUI is made visible.
function SegmentationFilteringPostprocessingProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)
processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',1);

% Parameter setup
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;


% set GUI with Parameters:

% For funParams.size_filters.xxx, tag names pattern as type_sf_parameterName, e.g. edit_sf_min_size
set(handles.edit_sf_min_size, 'String',num2str(funParams.size_filters.min_size))
set(handles.edit_sf_max_size_factor, 'String',num2str(funParams.size_filters.max_size_factor))
if funParams.size_filters.do_stats_filter
    set(handles.checkbox_sf_do_stats_filter, 'Value', 1)
else
    set(handles.checkbox_sf_do_stats_filter, 'Value', 0)
end

% For funParams.flow_consistency.xxx, tag names pattern as type_fc_parameterName, e.g. edit_fc_flow_threshold
set(handles.edit_fc_flow_threshold, 'String',num2str(funParams.flow_consistency.flow_threshold))
if funParams.flow_consistency.do_flow_remove
    set(handles.checkbox_fc_do_flow_remove, 'Value', 1)
else
    set(handles.checkbox_fc_do_flow_remove, 'Value', 0)
end
%Setup Distance Transform Method pop up menu (popupmenu_fc_dtform_method):
% The 'String' of popupmenu_fc_dtform_method does not match the funParams.flow_consistency.dtform_method
% So the 'String' of popupmenu_fc_dtform_method was set in the GUI's property inspector.
parVal = funParams.flow_consistency.dtform_method;
valSel  = find(ismember(SegmentationFilteringPostprocessingProcess.getValidDistTransMethod, parVal));
if isempty(valSel), valSel = 1; end
set(handles.popupmenu_fc_dtform_method, 'Value', valSel);

set(handles.edit_fc_power_dist, 'String',num2str(funParams.flow_consistency.power_dist))
set(handles.edit_fc_edt_fixed_point_percentile, 'String',num2str(funParams.flow_consistency.edt_fixed_point_percentile))
set(handles.edit_fc_smooth_skel_sigma, 'String',num2str(funParams.flow_consistency.smooth_skel_sigma))

set(handles.edit_fc_n_cpu, 'String',num2str(funParams.flow_consistency.n_cpu))

    % Set up the dependency of below parameters:
    % (1) gray out below params if step 2 is not ExternalSegment3DProcess - assume they use dtform_method='cellpose_improve' 
    iSegmentProc = userData.MD.getProcessIndex('ExternalSegment3DProcess', 'askUser', false, 'nDesired', Inf); % this can be multiple index
    if isempty(iSegmentProc)
        set([handles.text_fc_power_dist, handles.edit_fc_power_dist, ...
            handles.text_fc_edt_fixed_point_percentile, handles.edit_fc_edt_fixed_point_percentile, ...
            handles.text_fc_smooth_skel_sigma, handles.edit_fc_smooth_skel_sigma, ...
            handles.text_fc_dtform_method, handles.popupmenu_fc_dtform_method], 'Enable', 'off');
    end

    % (2) gray out Distance Transform Exponent if dtform_method is not 'cellpose_improve'
    if ~isequal(funParams.flow_consistency.dtform_method, 'cellpose_improve')
        set(handles.text_fc_power_dist, 'Enable','off');
        set(handles.edit_fc_power_dist, 'Enable','off');
    end
    % (3) gray out Euclidean Distance Transform Threshold Percentile if dtform_method is not 'cellpose_improve' nor 'fmm'
    if ~ismember(funParams.flow_consistency.dtform_method, {'cellpose_improve', 'fmm'})
        set(handles.text_fc_edt_fixed_point_percentile, 'Enable','off');
        set(handles.edit_fc_edt_fixed_point_percentile, 'Enable','off');
    end
    % (4) gray out Smooth Skeleton if dtform_method is not 'fmm_skel' nor 'cellpose_skel'
    if ~ismember(funParams.flow_consistency.dtform_method, {'fmm_skel', 'cellpose_skel'})
        set(handles.text_fc_smooth_skel_sigma, 'Enable','off');
        set(handles.edit_fc_smooth_skel_sigma, 'Enable','off');
    end


% Update user data and GUI data
handles.output = hObject;
set(handles.figure1, 'UserData', userData);
guidata(hObject, handles);





% --- Outputs from this function are returned to the command line.
function varargout = SegmentationFilteringPostprocessingProcessGUI_OutputFcn(hObject, eventdata, handles)
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
% eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
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

%  Check user input --------:
if isempty(get(handles.listbox_selectedChannels, 'String'))
   errordlg('Please select at least one input channel from ''Available Channels''.','Setting Error','modal') 
    return;
end

% For funParams.size_filters.xxx:
  % The Smallest Allowable Cell Size [voxels] is interger >0
if isnan(str2double(get(handles.edit_sf_min_size, 'String'))) ...
        || str2double(get(handles.edit_sf_min_size, 'String')) <= 0 ...
        || floor(str2double(get(handles.edit_sf_min_size, 'String'))) ~= str2double(get(handles.edit_sf_min_size, 'String'))
    errordlg('Please provide a valid input for ''The Smallest Allowable Cell Size [voxels]''.','Setting Error','modal');
    return;
end

  % Multiplier for Setting Maximum Cell Size is float >0
if isnan(str2double(get(handles.edit_sf_max_size_factor, 'String'))) ...
        || str2double(get(handles.edit_sf_max_size_factor, 'String')) <= 0
    errordlg('Please provide a valid input for ''Multiplier for Setting Maximum Cell Size''.','Setting Error','modal');
    return;
end

% For funParams.flow_consistency.xxx:
  % Maximum Mean Squared Error...  is float >0
if isnan(str2double(get(handles.edit_fc_flow_threshold, 'String'))) ...
        || str2double(get(handles.edit_fc_flow_threshold, 'String')) <= 0
    errordlg('Please provide a valid input for ''Maximum Mean Squared Error... ''.','Setting Error','modal');
    return;
end

  % below check only when the elements are not gray out:
if isequal(get(handles.edit_fc_power_dist, 'Enable'), 'on')
    % Distance Transform Exponent can be empty (None, string(missing) in Matlab) or float >0 and <=1
  if ~isempty(get(handles.edit_fc_power_dist, 'String')) && (str2double(get(handles.edit_fc_power_dist, 'String')) <= 0 ...
          || str2double(get(handles.edit_fc_power_dist, 'String')) > 1)
      errordlg('Please provide a valid input for ''Distance Transform Exponent''.','Setting Error','modal');
      return;
  end
end

if isequal(get(handles.edit_fc_edt_fixed_point_percentile, 'Enable'), 'on')
    % Euclidean Distance Transform Threshold Percentile is float >=0 and <=1
  if isnan(str2double(get(handles.edit_fc_edt_fixed_point_percentile, 'String'))) ...
          || str2double(get(handles.edit_fc_edt_fixed_point_percentile, 'String')) < 0 ...
          || str2double(get(handles.edit_fc_edt_fixed_point_percentile, 'String')) > 1
      errordlg('Please provide a valid input for ''Euclidean Distance Transform Threshold Percentile''.','Setting Error','modal');
      return;
  end
end

if isequal(get(handles.edit_fc_smooth_skel_sigma, 'Enable'), 'on')
    % Smooth Skeleton is float >=1
  if isnan(str2double(get(handles.edit_fc_smooth_skel_sigma, 'String'))) ...
          || str2double(get(handles.edit_fc_smooth_skel_sigma, 'String')) < 1
      errordlg('Please provide a valid input for ''Smooth Skeleton''.','Setting Error','modal');
      return;
  end
end

if isequal(get(handles.edit_fc_n_cpu, 'Enable'), 'on')
    % Number of CPU to Use (default is empty) can be empty (None, string(missing) in Matlab) or int >0
  if ~isempty(get(handles.edit_fc_n_cpu, 'String')) && (str2double(get(handles.edit_fc_n_cpu, 'String')) <= 0 ...
          || floor(str2double(get(handles.edit_fc_n_cpu, 'String'))) ~= str2double(get(handles.edit_fc_n_cpu, 'String')))
      errordlg('Please provide a valid input for ''Number of CPU to Use (default is empty)''.','Setting Error','modal');
      return;
  end
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

% For funParams.size_filters.xxx:
funParams.size_filters.min_size = int16(str2double(get(handles.edit_sf_min_size, 'String'))); % this parameter need to be a integer!
funParams.size_filters.max_size_factor = str2double(get(handles.edit_sf_max_size_factor, 'String'));
if get(handles.checkbox_sf_do_stats_filter, 'Value')
    funParams.size_filters.do_stats_filter = true;
else
    funParams.size_filters.do_stats_filter = false;
end

% For funParams.flow_consistency.xxx:
funParams.flow_consistency.flow_threshold = str2double(get(handles.edit_fc_flow_threshold, 'String'));
if get(handles.checkbox_fc_do_flow_remove, 'Value')
    funParams.flow_consistency.do_flow_remove = true;
else
    funParams.flow_consistency.do_flow_remove = false;
end

  % Below Retrieve GUI-defined parameters in this panel when the elements are not gray out:
if isequal(get(handles.popupmenu_fc_dtform_method, 'Enable'), 'on')
  selType = get(handles.popupmenu_fc_dtform_method, 'Value'); 
  funParams.flow_consistency.dtform_method = SegmentationFilteringPostprocessingProcess.getValidDistTransMethod{selType};
end

if isequal(get(handles.edit_fc_power_dist, 'Enable'), 'on')
    % this can be empty (None, string(missing) in Matlab) or float:
  if isempty(get(handles.edit_fc_power_dist, 'String'))
      funParams.flow_consistency.power_dist = string(missing);
  else
      funParams.flow_consistency.power_dist = str2double(get(handles.edit_fc_power_dist, 'String'));
  end
end

if isequal(get(handles.edit_fc_edt_fixed_point_percentile, 'Enable'), 'on')
  funParams.flow_consistency.edt_fixed_point_percentile = str2double(get(handles.edit_fc_edt_fixed_point_percentile, 'String'));
end

if isequal(get(handles.edit_fc_smooth_skel_sigma, 'Enable'), 'on')
  funParams.flow_consistency.smooth_skel_sigma = str2double(get(handles.edit_fc_smooth_skel_sigma, 'String'));
end

if isequal(get(handles.edit_fc_n_cpu, 'Enable'), 'on')
    % this can be empty (None, string(missing) in Matlab) or integer:
  if isempty(get(handles.edit_fc_n_cpu, 'String'))
      funParams.flow_consistency.n_cpu = string(missing);
  else
      funParams.flow_consistency.n_cpu = int16(str2double(get(handles.edit_fc_n_cpu, 'String'))); % this parameter need to be a integer!
  end
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


% --- Executes on selection change in popupmenu_fc_dtform_method.
function popupmenu_fc_dtform_method_Callback(hObject, eventdata, handles)
% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_fc_dtform_method contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_fc_dtform_method
% Default all controls to 'off'
set([handles.text_fc_power_dist, handles.edit_fc_power_dist, ...
     handles.text_fc_edt_fixed_point_percentile, handles.edit_fc_edt_fixed_point_percentile, ...
     handles.text_fc_smooth_skel_sigma, handles.edit_fc_smooth_skel_sigma], 'Enable', 'off');

% set some controls to 'on' in below conditions
switch get(hObject, 'Value')
    case 1
        set([handles.text_fc_power_dist, handles.edit_fc_power_dist, ...
             handles.text_fc_edt_fixed_point_percentile, handles.edit_fc_edt_fixed_point_percentile], 'Enable', 'on');
    case 3
        set([handles.text_fc_edt_fixed_point_percentile, handles.edit_fc_edt_fixed_point_percentile], 'Enable', 'on');
    case {4, 5}
        set([handles.text_fc_smooth_skel_sigma, handles.edit_fc_smooth_skel_sigma], 'Enable', 'on');
end

