function varargout = msaSegmentationProcessGUI(varargin)
%MSASEGMENTATIONPROCESSGUI MATLAB code file for msaSegmentationProcessGUI.fig
%      MSASEGMENTATIONPROCESSGUI, by itself, creates a new MSASEGMENTATIONPROCESSGUI or raises the existing
%      singleton*.
%
%      H = MSASEGMENTATIONPROCESSGUI returns the handle to a new MSASEGMENTATIONPROCESSGUI or the handle to
%      the existing singleton*.
%
%      MSASEGMENTATIONPROCESSGUI('Property','Value',...) creates a new MSASEGMENTATIONPROCESSGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to msaSegmentationProcessGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      MSASEGMENTATIONPROCESSGUI('CALLBACK') and MSASEGMENTATIONPROCESSGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in MSASEGMENTATIONPROCESSGUI.M with the given input
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

% Edit the above text to modify the response to help msaSegmentationProcessGUI

% Last Modified by GUIDE v2.5 18-Aug-2021 14:49:29

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @msaSegmentationProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @msaSegmentationProcessGUI_OutputFcn, ...
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


% --- Executes just before msaSegmentationProcessGUI is made visible.
function msaSegmentationProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)


processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',1);

% Parameter setup
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;


% set GUI with Parameters

if funParams.tightness == -1
    handles.tightness_checkbox.Value = 0;
    handles.tightness_slider.Enable = 'off';
    handles.tightness_slider.Value = .5;
    handles.tightness_display.String = 'Inactive';
    handles.tightness_display.Enable = 'off';
elseif funParams.tightness <=1 && funParams.tightness >= 0
    handles.tightness_checkbox.Value = 1;
    handles.tightness_slider.Enable = 'on';
    handles.tightness_slider.Value = funParams.tightness;
    handles.tightness_display.String = num2str(handles.tightness_slider.Value);
end

if funParams.numVotes == -1
    handles.numVotes_checkbox.Value = 0;
    handles.numVotes_slider.Enable = 'off';
    handles.numVotes_slider.Value = 22;
    handles.numVotes_display.String = 'Inactive';
    handles.numVotes_display.Enable = 'off';
elseif funParams.numVotes <=42 && funParams.numVotes >= 0
    handles.numVotes_checkbox.Value = 1;
    handles.numVotes_slider.Enable = 'on';
    handles.numVotes_slider.Value = funParams.numVotes;
    handles.numVotes_display.String = num2str(handles.numVotes_slider.Value);
end

set(handles.edit_ObjectNumber, 'String',num2str(funParams.ObjectNumber))
set(handles.edit_finalRefinementRadius, 'String',num2str(funParams.finalRefinementRadius))

% set GUI with popupmenu_ProcessIndex
sumChanProc =  cellfun(@(x) isa(x,'GenerateSummationChannelProcess'),userData.MD.processes_);
sumChanProcID=find(sumChanProc);
sumChanProcNames = cellfun(@(x) x.getName(),userData.MD.processes_(sumChanProc),'Unif',false);
sumChanProcString = vertcat('Choose later',sumChanProcNames(:));
sumChanProcData=horzcat({[]},num2cell(sumChanProcID));
sumChanProcValue = find(cellfun(@(x) isequal(x,funParams.ProcessIndex),sumChanProcData));
if isempty(sumChanProcValue), sumChanProcValue = 1; end
set(handles.popupmenu_ProcessIndex,'String',sumChanProcString,...
    'UserData',sumChanProcData,'Value',sumChanProcValue);

% Update channels listboxes depending on the selected process
popupmenu_ProcessIndex_Callback(hObject, eventdata, handles)

if funParams.useSummationChannel == 1
    handles.text_GenerateSummationChannelProcess.Enable = 'on';
    handles.popupmenu_ProcessIndex.Enable = 'on';
elseif funParams.useSummationChannel == 0
    handles.text_GenerateSummationChannelProcess.Enable = 'off';
    handles.popupmenu_ProcessIndex.Enable = 'off';
end

% Update user data and GUI data
handles.output = hObject;
set(handles.figure1, 'UserData', userData);
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = msaSegmentationProcessGUI_OutputFcn(hObject, eventdata, handles)
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


%  Check user input --------
if isempty(get(handles.listbox_selectedChannels, 'String'))
    errordlg('Please select at least one input channel from ''Available Channels''.','Setting Error','modal')
    return;
end

if isnan(str2double(get(handles.edit_ObjectNumber, 'String'))) ...
    || str2double(get(handles.edit_ObjectNumber, 'String')) < 0
  errordlg('Please provide a valid input for ''Objec Number''.','Setting Error','modal');
  return;
end

if isnan(str2double(get(handles.edit_finalRefinementRadius, 'String'))) ...
    || str2double(get(handles.edit_finalRefinementRadius, 'String')) < 0
  errordlg('Please provide a valid input for ''Final Refinement Radius''.','Setting Error','modal');
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

funParams.ObjectNumber = str2double(get(handles.edit_ObjectNumber, 'String'));
funParams.finalRefinementRadius = str2double(get(handles.edit_finalRefinementRadius, 'String'));

if handles.tightness_checkbox.Value == 1
    funParams.tightness = str2double(handles.tightness_display.String);
else
    funParams.tightness = -1;
end

if handles.numVotes_checkbox.Value == 1
    funParams.numVotes = str2double(handles.numVotes_display.String);
else
    funParams.numVotes = -1;
end

% Retrieve GenerateSummationChannelProcess index
if handles.checkbox_useSummationChannel.Value == 1
  funParams.useSummationChannel = 1;
  props=get(handles.popupmenu_ProcessIndex,{'UserData','Value'});
  funParams.ProcessIndex = props{1}{props{2}};
else 
  funParams.useSummationChannel = 0;
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

% --- Executes on button press in useSummationChannel_checkbox.
function checkbox_useSummationChannel_Callback(hObject, eventdata, handles)
% hObject    handle to useSummationChannel_checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of useSummationChannel_checkbox
if handles.checkbox_useSummationChannel.Value == 0
    handles.text_GenerateSummationChannelProcess.Enable = 'off';
    handles.popupmenu_ProcessIndex.Enable = 'off';
else
    handles.text_GenerateSummationChannelProcess.Enable = 'on';
    handles.popupmenu_ProcessIndex.Enable = 'on';
end

% --- Executes on button press in tightness_checkbox.
function tightness_checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to tightness_checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tightness_checkbox
if handles.tightness_checkbox.Value == 0
    handles.tightness_slider.Enable = 'off';
    handles.tightness_display.String = 'Inactive';
    handles.tightness_display.Enable = 'off';
    
    handles.numVotes_checkbox.Value = 1;
    handles.numVotes_slider.Enable = 'on';
    handles.numVotes_display.String = num2str(handles.numVotes_slider.Value);
    handles.numVotes_display.Enable = 'on';
else
    handles.tightness_slider.Enable = 'on';
    handles.tightness_display.String = num2str(handles.tightness_slider.Value);
    handles.tightness_display.Enable = 'on';

    handles.numVotes_checkbox.Value = 0;
    handles.numVotes_slider.Enable = 'off';
    handles.numVotes_display.String = 'Inactive';
    handles.numVotes_display.Enable = 'off';
end

% --- Executes on button press in numVotes_checkbox.
function numVotes_checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to numVotes_checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of numVotes_checkbox
if handles.numVotes_checkbox.Value == 0
    handles.numVotes_slider.Enable = 'off';
    handles.numVotes_display.String = 'Inactive';
    handles.numVotes_display.Enable = 'off';

    handles.tightness_checkbox.Value = 1;
    handles.tightness_slider.Enable = 'on';
    handles.tightness_display.String = num2str(handles.tightness_slider.Value);
    handles.tightness_display.Enable = 'on';
else
    handles.numVotes_slider.Enable = 'on';
    handles.numVotes_display.String = num2str(handles.numVotes_slider.Value);
    handles.numVotes_display.Enable = 'on';

    handles.tightness_checkbox.Value = 0;
    handles.tightness_slider.Enable = 'off';
    handles.tightness_display.String = 'Inactive';
    handles.tightness_display.Enable = 'off';
end

% --- Executes on selection change in popupmenu_ProcessIndex.
function popupmenu_ProcessIndex_Callback(hObject, eventdata, handles)
% Retrieve selected process ID
props= get(handles.popupmenu_ProcessIndex,{'UserData','Value'});
procID = props{1}{props{2}};

% Read process and check available channels
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
if isempty(procID)
    allChannelIndex=1:numel(userData.MD.channels_);
else
    allChannelIndex = find(userData.MD.processes_{procID}.checkChannelOutput);
end

% Set up available channels listbox
if ~isempty(allChannelIndex)
    if isempty(procID)
        channelString = userData.MD.getChannelPaths(allChannelIndex);
    else
        channelString = userData.MD.processes_{procID}.outFilePaths_(1,allChannelIndex);
    end
else
    channelString = {};
end
set(handles.listbox_availableChannels,'String',channelString,'UserData',allChannelIndex);

% Set up selected channels listbox
channelIndex = get(handles.listbox_selectedChannels, 'UserData');
channelIndex = intersect(channelIndex,allChannelIndex);
if ~isempty(channelIndex)
    if isempty(procID)
        channelString = userData.MD.getChannelPaths(channelIndex);
    else
        channelString = userData.MD.processes_{procID}.outFilePaths_(1,channelIndex);
    end
else
    channelString = {};
end
set(handles.listbox_selectedChannels,'String',channelString,'UserData',channelIndex);




% --- Executes on slider movement.
function tightness_slider_Callback(hObject, eventdata, handles)
% hObject    handle to tightness_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles.tightness_display.String = num2str(round((handles.tightness_slider.Value)*10)/10); % round tightness to 0.1 per step when drag slider


% --- Executes during object creation, after setting all properties.
function tightness_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tightness_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
handles.tightness_slider.Value = .5;



% --- Executes on numVotes slider movement.
function numVotes_slider_Callback(hObject, eventdata, handles)
% hObject    handle to numVotes_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles.numVotes_display.String = num2str(round(handles.numVotes_slider.Value)); % round numVotes to 1 per step when drag slider


% --- Executes during object creation, after setting all properties.
function numVotes_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numVotes_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
handles.numVotes_slider.Value = 22;