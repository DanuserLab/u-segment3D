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
% Copyright (C) 2025, Danuser Lab - UTSouthwestern 
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

processGUI_OpeningFcn(hObject, eventdata, handles, varargin{:},'initChannel',0); % change initChannel from 1 to 0, so it will local checkbox_all_Callback, pushbutton_select_Callback, and pushbutton_delete_Callback.

% Parameter setup
userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end
funParams = userData.crtProc.funParams_;

% Below line 59-91 is from processGUI_OpeningFcn, b/c initChannel is 0, then this part will not be called
% in processGUI_OpeningFcn, so need to add here.
% Set up available input channels
set(handles.listbox_availableChannels,'String',userData.MD.getChannelPaths(), ...
    'UserData',1:numel(userData.MD.channels_));

channelIndex = funParams.ChannelIndex;

% Find any parent process
userData.parentProc = userData.crtPackage.getParent(userData.procID);
if numel(userData.parentProc) > 1
    userData.parentProc = userData.parentProc(end); % use last parentProc as parentProc
end
if isempty(userData.crtPackage.processes_{userData.procID}) && ~isempty(userData.parentProc)
    % Check existence of all parent processes
    emptyParentProc = any(cellfun(@isempty,userData.crtPackage.processes_(userData.parentProc)));
    if ~emptyParentProc
        % Intersect channel index with channel index of parent processes
        parentChannelIndex = @(x) userData.crtPackage.processes_{x}.funParams_.ChannelIndex;
        for i = userData.parentProc
            channelIndex = intersect(channelIndex,parentChannelIndex(i));
        end
    end
   
end


if ~isempty(channelIndex)
    channelString = userData.MD.getChannelPaths(channelIndex);
else
    channelString = {};
end

set(handles.listbox_selectedChannels,'String',channelString,...
    'UserData',channelIndex);

% By default the avail channel list box always list the raw images (see line 60). But preview alway shows the image from the last valid parentProc. So, we do not have to do below for biosensorsPackage.
% % In BiosensorsPackage, auto-set listbox_availableChannels and listbox_selectedChannels to ShadeCorrectionProcess or CropShadeCorrectROIProcess's output
% if isa(userData.crtPackage, 'BiosensorsPackage')
%     for i = numel(userData.crtProc.processTree_):-1:1
%         if isa(userData.crtProc.processTree_{i}, 'CropShadeCorrectROIProcess')
%             processIndex = i;
%             break; % Exit the loop as the CropShadeCorrectROIProcess is found
%         elseif isa(userData.crtProc.processTree_{i}, 'ShadeCorrectionProcess')
%             processIndex = i;
%         end
%     end
% 
%     % Update avail input channels and selected channels:
%     % Read process and check available channels
%     if isempty(processIndex)
%         allChannelIndex=1:numel(userData.MD.channels_);
%     else
%         allChannelIndex = find(userData.MD.processes_{processIndex}.checkChannelOutput);
%     end
% 
%     % Set up available channels listbox
%     if ~isempty(allChannelIndex)
%         if isempty(processIndex)
%             channelString = userData.MD.getChannelPaths(allChannelIndex);
%         else
%             channelString = userData.MD.processes_{processIndex}.outFilePaths_(1,allChannelIndex);
%         end
%     else
%         channelString = {};
%     end
%     set(handles.listbox_availableChannels,'String',channelString,'UserData',allChannelIndex);
% 
%     % Set up selected channels listbox
%     channelIndex = get(handles.listbox_selectedChannels, 'UserData');
%     channelIndex = intersect(channelIndex,allChannelIndex);
%     if ~isempty(channelIndex)
%         if isempty(processIndex)
%             channelString = userData.MD.getChannelPaths(channelIndex);
%         else
%             channelString = userData.MD.processes_{processIndex}.outFilePaths_(1,channelIndex);
%         end
%     else
%         channelString = {};
%     end
%     set(handles.listbox_selectedChannels,'String',channelString,'UserData',channelIndex);
% end



% set GUI with Parameters

if funParams.tightness == -1
    handles.tightness_checkbox.Value = 0;
    handles.tightness_slider.Enable = 'off';
    handles.tightness_slider.Value = .50;
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

if isequal(funParams.figVisible, 'on')
  set(handles.checkbox_figFlag, 'Value', 1)
else
  set(handles.checkbox_figFlag, 'Value', 0)
end

if isequal(funParams.verbose, 'on')
  set(handles.checkbox_verbose, 'Value', 1)
else
  set(handles.checkbox_verbose, 'Value', 0)
end

% Initialize previewing constants
userData.previewFig =-1;
userData.chanIndx = 0;
userData.imIndx=0;
userData.processIndex=funParams.ProcessIndex;
set(hObject, 'UserData', userData);


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
set(handles.checkbox_useSummationChannel,'Value',funParams.useSummationChannel);

% Update channels listboxes depending on the selected process
popupmenu_ProcessIndex_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData'); % retrive the userData again b/c it may be just updated in the callback above.

if funParams.useSummationChannel == 1
    handles.text_GenerateSummationChannelProcess.Enable = 'on';
    handles.popupmenu_ProcessIndex.Enable = 'on';
elseif funParams.useSummationChannel == 0
    handles.text_GenerateSummationChannelProcess.Enable = 'off';
    handles.popupmenu_ProcessIndex.Enable = 'off';
end


% Initialize the frame number slider and edit for preview panel
nFrames=userData.MD.nFrames_;
if nFrames > 1
    set(handles.slider_frameNumber,'Value',1,'Min',1,...
        'Max',nFrames,'SliderStep',[1/double(nFrames)  10/double(nFrames)]);
else
    set(handles.slider_frameNumber,'Enable','off');
end
set(handles.text_nFrames,'String',['/ ' num2str(nFrames)]);
set(handles.edit_frameNumber,'Value',1);


% Update user data and GUI data
handles.output = hObject;
set(hObject, 'UserData', userData);
guidata(hObject, handles);
update_data(hObject,eventdata,handles);


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

% close preview figure and delete preview folder
if ishandle(userData.previewFig), delete(userData.previewFig); end 
PreviewOutputDir = [handles.figure1.UserData.MD.outputDirectory_ filesep 'MSApreviewTempDir'];
if isfolder(PreviewOutputDir); rmdir(PreviewOutputDir, 's'); end

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

% delete preview folder
PreviewOutputDir = [handles.figure1.UserData.MD.outputDirectory_ filesep 'MSApreviewTempDir'];
if isfolder(PreviewOutputDir); rmdir(PreviewOutputDir, 's'); end
    
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

if get(handles.checkbox_figFlag, 'Value') == 1
    funParams.figVisible = 'on';
else
    funParams.figVisible = 'off';
end

if get(handles.checkbox_verbose, 'Value') == 1
    funParams.verbose = 'on';
else
    funParams.verbose = 'off';
end

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
  funParams.ProcessIndex = [];
end

% close preview figure and delete preview folder
if ishandle(userData.previewFig), delete(userData.previewFig); end 
PreviewOutputDir = [handles.figure1.UserData.MD.outputDirectory_ filesep 'MSApreviewTempDir'];
if isfolder(PreviewOutputDir); rmdir(PreviewOutputDir, 's'); end

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
%     handles.popupmenu_ProcessIndex.Value = 1; % do not do this, b/c Input Channels list boxes will not update.
else
    handles.text_GenerateSummationChannelProcess.Enable = 'on';
    handles.popupmenu_ProcessIndex.Enable = 'on';
end

% update_data(hObject,eventdata,handles); % No need to update preview, b/c preview alway shows the image from the last valid parentProc


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

update_data(hObject,eventdata,handles);


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

update_data(hObject,eventdata,handles);


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


% update_data(hObject,eventdata,handles); % No need to update preview, b/c preview alway shows the image from the last valid parentProc




% --- Executes on slider movement.
function tightness_slider_Callback(hObject, eventdata, handles)
% hObject    handle to tightness_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles.tightness_display.String = num2str(round((handles.tightness_slider.Value)*100)/100); % round tightness to 0.01 per step when drag slider

update_data(hObject,eventdata,handles);


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

if round(handles.numVotes_slider.Value) == 0 % avoid numVotes to be 0, otherwise error.
    set(handles.numVotes_slider, 'Value', 1);
    handles.numVotes_display.String = '1';
end

update_data(hObject,eventdata,handles);


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

% --- Executes on button press in checkbox_preview.
function checkbox_preview_Callback(hObject, eventdata, handles)

if get(handles.checkbox_preview,'Value'), 
    update_data(hObject,eventdata,handles); 
else
    userData = get(handles.figure1, 'UserData');
    if ishandle(userData.previewFig), delete(userData.previewFig); end
    % Save data and update graphics
    set(handles.figure1,'UserData',userData);
    guidata(hObject, handles);
end

function imageNumber_edition(hObject,eventdata, handles)

% Retrieve the value of the selected image
if strcmp(get(hObject,'Tag'),'edit_frameNumber')
    imageNumber = str2double(get(handles.edit_frameNumber, 'String'));
else
    imageNumber = get(handles.slider_frameNumber, 'Value');
end
imageNumber=round(imageNumber);

% Check the validity of the supplied threshold
if isnan(imageNumber) || imageNumber < 0 || imageNumber > get(handles.slider_frameNumber,'Max')
    warndlg('Please provide a valid frame number.','Setting Error','modal');
end

set(handles.slider_frameNumber,'Value',imageNumber);
set(handles.edit_frameNumber,'String',imageNumber);

% Save data and update graphics
guidata(hObject, handles);
update_data(hObject,eventdata,handles);


function edit_ObjectNumber_Callback(hObject, eventdata, handles)

update_data(hObject,eventdata,handles);


function edit_finalRefinementRadius_Callback(hObject, eventdata, handles)

update_data(hObject,eventdata,handles);


% Normally, below 3 callback fcns were in processGUI_OpeningFcn, however, here we need to update_data for the preview, 
% so we need to add checkbox_all_Callback, pushbutton_select_Callback, and pushbutton_delete_Callback here. - QZ
% --- Executes on button press in checkbox_all.
function checkbox_all_Callback(hObject, eventdata, handles)
% Retrieve available channels properties
availableProps = get(handles.listbox_availableChannels, {'String','UserData'});
if isempty(availableProps{1}), return; end

% Update selected channels
if get(hObject,'Value')
    set(handles.listbox_selectedChannels, 'String', availableProps{1},...
        'UserData',availableProps{2});
else
    set(handles.listbox_selectedChannels, 'String', {}, 'UserData',[], 'Value',1);
end

update_data(hObject,eventdata,handles);


% --- Executes on button press in pushbutton_select.
function pushbutton_select_Callback(hObject, eventdata, handles)
% Retrieve  channels properties
availableProps = get(handles.listbox_availableChannels, {'String','UserData','Value'});
selectedProps = get(handles.listbox_selectedChannels, {'String','UserData'});

% Find new elements and set them to the selected listbox
newID = availableProps{3}(~ismember(availableProps{1}(availableProps{3}),selectedProps{1}));
selectedChannels = horzcat(selectedProps{1}',availableProps{1}(newID)');
selectedData = horzcat(selectedProps{2}, availableProps{2}(newID));
set(handles.listbox_selectedChannels, 'String', selectedChannels, 'UserData', selectedData);

update_data(hObject,eventdata,handles);


% --- Executes on button press in pushbutton_delete.
function pushbutton_delete_Callback(hObject, eventdata, handles)
% Generic callback to be exectuted when a selected channel is removed from
% the graphical settings interface

% Get selected properties and returin if empty
selectedProps = get(handles.listbox_selectedChannels, {'String','UserData','Value'});
if isempty(selectedProps{1}) || isempty(selectedProps{3}),return; end

% Delete selected item
selectedProps{1}(selectedProps{3}) = [ ];
selectedProps{2}(selectedProps{3}) = [ ];
set(handles.listbox_selectedChannels, 'String', selectedProps{1},'UserData',selectedProps{2},...
    'Value',max(1,min(selectedProps{3},numel(selectedProps{1}))));

update_data(hObject,eventdata,handles);



function update_data(hObject,eventdata, handles)

userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end


% Retrieve the channex index, frame number, and process index
props=get(handles.listbox_selectedChannels,{'UserData','Value'});
if isempty(props{1}), return; end
chanIndx = props{1}(props{2});
imIndx = get(handles.slider_frameNumber,'Value');

% Load a new image in case the image number or channel has been changed
% NOTE: preview always shows the image from the last valid parentProc. - QZ
if (chanIndx~=userData.chanIndx) ||  (imIndx~=userData.imIndx)
    if ~isempty(userData.parentProc) && ~isempty(userData.crtPackage.processes_{userData.parentProc}) &&...
            userData.crtPackage.processes_{userData.parentProc}.checkChannelOutput(chanIndx)
        userData.imData=userData.crtPackage.processes_{userData.parentProc}.loadOutImage(chanIndx,imIndx);
    else
        userData.imData=userData.MD.channels_(chanIndx).loadImage(imIndx);
    end

    userData.updateImage=1;
    userData.chanIndx=chanIndx;
    userData.imIndx=imIndx;
else
    userData.updateImage=0;
end


% Save the data
set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);

% Update graphics if applicable
if get(handles.checkbox_preview,'Value') % Preview

    imData=userData.imData;

    %% Algorithm for preview, adapted from multiScaleAutoSeg_multiObject.m
    % see MSA_Seg_multiObject_imDir.m
    % Edit to make it work for all MD.Reader, such as BioFormatsReader. Before, the algorithm only works for TiffSeriesReader.
    % Edit again to make it also work when input is from output of a previous process. - Qiongjing (Jenny) Zou, Nov 2022

    k = chanIndx;
    PreviewOutputDir = [userData.MD.outputDirectory_ filesep 'MSApreviewTempDir'];
    if ~isfolder(PreviewOutputDir); mkdir(PreviewOutputDir); end % deleted this when setting GUI closed
    masksOutDir = [PreviewOutputDir filesep 'Masks'];
    if ~isfolder(masksOutDir); mkdir(masksOutDir); end

    I = cell(1, 1);
    I{1} = imData;
    imgStack = imData;

    % Retreive Parameters on the GUI:
    % tightness and numVotes are exclusive options: if one is chosen, the other is inactive (-1);
    if handles.tightness_checkbox.Value == 1
        currTightness = str2double(handles.tightness_display.String);
    else
        currTightness = -1;
    end
    if handles.numVotes_checkbox.Value == 1
        currNumVotes = str2double(handles.numVotes_display.String);
    else
        currNumVotes = -1;
    end
    currObjectNumber = str2double(get(handles.edit_ObjectNumber, 'String'));
    currFinalRefinementRadius = str2double(get(handles.edit_finalRefinementRadius, 'String'));
    % below params not on GUI:
    currImagesOut = 1; % default value
    currFigVisible = 'on'; % always on for preview
    currVerbose = 'off'; % always off for preview
    currMinimumSize = 10; % default value

    % call the main algorithm fcn:
    refinedMask = MSA_Seg_multiObject_imDir_2(I, imgStack, 1, PreviewOutputDir, ...
        masksOutDir, k, imIndx, 'tightness', currTightness, 'numVotes', currNumVotes, ...
        'ObjectNumber', currObjectNumber, 'finalRefinementRadius', currFinalRefinementRadius, ...
        'imagesOut', currImagesOut, 'figVisible', currFigVisible, 'MinimumSize', currMinimumSize, 'verbose', currVerbose);

    %%%% end of algorithm

    %% imagesOut

    if currNumVotes >= 0
        prefname = ['numVotes_', num2str(currNumVotes)];
    elseif currTightness >= 0
        prefname = ['tightness_', num2str(currTightness)];
    else
        prefname = '_';
    end

    dName2 = ['MSASeg_maskedImages_' prefname '_for_channel_' num2str(k) '_frame_' num2str(imIndx)];
    imOutDir = fullfile(PreviewOutputDir, dName2);
    if ~isdir(imOutDir); mkdir(imOutDir); end

    allint = imgStack(:);
    intmin = quantile(allint, 0.002);
    intmax = quantile(allint, 0.998);

    % Create figure if non-existing or closed
    if ~ishandle(userData.previewFig)
        userData.previewFig = figure('NumberTitle','off','Name','MSA segmentation preview',...
            'Position',[50 50 userData.MD.imSize_(2) userData.MD.imSize_(1)]);
        axes('Position',[0 0 1 1]);
    else
        figure(userData.previewFig);
    end

    for fr = 1 % preview 1 frame
        imshow(I{fr}, [intmin, intmax])
        hold on
        bdd = bwboundaries(refinedMask{fr});

        for k = 1:numel(bdd)
            bdd1 = bdd{k};
            plot(bdd1(:,2), bdd1(:,1), 'r');
        end
        hold off

        h = getframe(userData.previewFig);  % Capture the frame from the figure
        imwrite(h.cdata, [imOutDir, filesep 'frame_', num2str(imIndx), '.tif'])
    end


    set(handles.figure1, 'UserData', userData);
    guidata(hObject,handles);
end
% end of update_data fcn


function refinedMask = MSA_Seg_multiObject_imDir_2(I, imgStack, frmax, outputDir, masksOutDir, iChan, imIndx, varargin)
% deleted input arg fileNames, as we do not need to save the file for preview
% added imIndx (frameNumber) input for preview
% frmax is just 1 for preview
% temperary put outputDir as userData.MD.outputDirectory_ and created a temp dir in it for masksOutDir

%% Parse input

ip = inputParser;
ip.addParameter('tightness', 0.5, @(x) isnumeric(x) && (x==-1 || x >= 0 || x<=1));
ip.addParameter('numVotes', -1);
ip.addParameter('imagesOut', 1);
ip.addParameter('figVisible', 'on');
ip.addParameter('finalRefinementRadius', 1);
ip.addParameter('MinimumSize', 10);
ip.addParameter('ObjectNumber', 1000);
ip.addParameter('verbose', 'off');
%ip.addParameter('parpoolNum', 1);

ip.parse(varargin{:});
p = ip.Results;

if (p.numVotes > 0); p.tightness = -1; end

%% control parameter structs to prevent redundant running when only a threshold is changed

p1 = p;     % MSA seg parameter struct

% Select MSA seg parameters except the threshold parameters
p2 = struct();
p2.finalRefinementRadius = p.finalRefinementRadius;
p2.MinimumSize = p.MinimumSize;
p2.ObjectNumber = p.ObjectNumber;

% Get old parameter
if isfile([outputDir filesep 'p2_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.mat'] )
    tmp = load([outputDir filesep 'p2_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.mat']);
    old_p2 = tmp.p2;
else
    % If it is the 1st run (w/o 'p2.mat' output), then make a fake 'old_p2'
    % struct, which always differs from 'p2' to run the segmentation in the below.
    old_p2 = struct();
    old_p2.finalRefinementRadius = -1;       % a fake value
end

if ~isfolder(outputDir); mkdir(outputDir); end
save([outputDir filesep 'p1_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.mat'], 'p1');
save([outputDir filesep 'p2_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.mat'], 'p2');

%% -------- Parameters ---------- %%

if ~isdir(masksOutDir); mkdir(masksOutDir); end

pString = 'MSA_mask_';      %Prefix for saving masks to file


%% Run MSA seg only if it has not run with the same parameter except threshodling parms
% Check if MSA Seg is once run for this movieData.
% Only if not, run MSA algorithm to compute voting score Array (step 1) and
% masks (step 2).
% If previous results for the same parameters (except thresholds) exist,
% then compute only masks (step 2).

scoreArrayFilePath = [outputDir filesep 'scoreArray_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.mat'];
if ~isfile(scoreArrayFilePath) || ~isequaln(p2, old_p2)

    [refinedMask, voteScoreImgs] = MSA_Seg_1stRun(p, outputDir, frmax, imgStack, iChan, imIndx, p.verbose);

    % voteScoreImg
    % dir name for vote score images
    imOutDir2 = [outputDir filesep 'MSASeg_voteScoreImgs_for_channel_' num2str(iChan) '_frame_' num2str(imIndx)];
    if ~isfolder(imOutDir2); mkdir(imOutDir2); end

    for fr = frmax % preview frame number
        imwrite(voteScoreImgs{fr}, fullfile(imOutDir2, ['voteScores_', 'frame_' num2str(imIndx),'.tif']) );
    end

else

    refinedMask = MSA_Seg_2ndRun(p, outputDir, iChan, imIndx, p.verbose);

end

%% save mask images

for fr = frmax % preview frame number
    %Write the refined mask to file
    imwrite(mat2gray(refinedMask{fr}), fullfile(masksOutDir, [pString, '_frame_' num2str(imIndx),'.tif']) );
end


% end of MSA_Seg_multiObject_imDir_2 fcn



%% When MSA seg is run for the first time with the same segmentation parameter

function [refinedMask, voteScoreImgs] = MSA_Seg_1stRun(p, outputDir, frmax, imgStack, iChan, imIndx, verbose)
    %% Time series of 5 numbers

    pixelmat = reshape(imgStack, [], frmax);
    pixelmat1 = pixelmat;
    pixelmat1(pixelmat1 == 0) = NaN;
    %sum(isnan(pixelmat1(:)))

    mts = mean(pixelmat1, 1, 'omitnan');
    medts = median(pixelmat1, 1, 'omitnan');
    q1ts = quantile(pixelmat1, 0.25, 1);
    q3ts = quantile(pixelmat1, 0.75, 1);
    q99ts = quantile(pixelmat1, 0.99, 1);
    q01ts = quantile(pixelmat1, 0.01, 1);

    fts = figure('Visible', 'off'); % turn off stat fig for preview
    plot(mts)
    hold on

    plot(medts)
    plot(q1ts)
    plot(q3ts)
    plot(q01ts)
    plot(q99ts)
    hold off

    legend('Mean', 'Median', 'Perct25', 'Perct75', 'Perct01', 'Perct99')
    title('Time series of 5 summary statistics')

%     %% saveas 
%     saveas(fts, [outputDir filesep 'TS_of_5statistics_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.png'], 'png')
    
    %% Multi Scale Segmentation

    refinedMask = cell(frmax, 1);
    voteScoreImgs = cell(frmax, 1); 
    currTightness = p.tightness;
    currNumVotes = p.numVotes;
    
    scoreArray = zeros(size(imgStack));    

    for fr = frmax % preview frame number
        disp('=====')
        disp('Preview Frame. Please wait...')    
        im = imgStack(:,:,fr);
        [refinedMask{fr}, voteScoreImgs{fr}, scoreArray(:,:,fr)] = ...
            multiscaleSeg_multiObject_im(im, ...
                'tightness', currTightness, 'numVotes', currNumVotes, ...
                'finalRefinementRadius', p.finalRefinementRadius, ...
                'MinimumSize', p.MinimumSize, 'ObjectNumber', p.ObjectNumber, 'verbose', verbose);
    end
    
    %% save voting scoreArray
    %save(fullfile(outputDir, 'scoreArray.mat'), 'scoreArray');

    save([outputDir filesep 'scoreArray_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.mat'], 'scoreArray') % important, once created will use for diff threshold


% end of MSA_Seg_1stRun fcn



%% When MSA seg is already run with the same parameters except thresholds

function refinedMask = MSA_Seg_2ndRun(p, outputDir, iChan, imIndx, verbose)
    
    % Load scoreArray    
    tmp = load([outputDir filesep 'scoreArray_for_channel_' num2str(iChan) '_frame_' num2str(imIndx) '.mat']);
    scoreArray = tmp.scoreArray;

    refinedMask = multiscaleSeg_multiObject_2ndRun(scoreArray, ...
            'numVotes', p.numVotes, 'tightness', p.tightness, ...
            'finalRefinementRadius', p.finalRefinementRadius, ...
            'MinimumSize', p.MinimumSize, 'ObjectNumber', p.ObjectNumber, 'verbose', verbose);
% end of MSA_Seg_2ndRun fcn