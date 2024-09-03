function varargout = cropMovieGUI(varargin)
% cropMovieGUI M-file for cropMovieGUI.fig
%      cropMovieGUI, by itself, creates a new cropMovieGUI or raises the existing
%      singleton*.
%
%      H = cropMovieGUI returns the handle to a new cropMovieGUI or the handle to
%      the existing singleton*.
%
%      cropMovieGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in cropMovieGUI.M with the given input arguments.
%
%      cropMovieGUI('Property','Value',...) creates a new cropMovieGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before cropMovieGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to cropMovieGUI_OpeningFcn via varargin.
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

% Edit the above text to modify the response to help cropMovieGUI

% Last Modified by GUIDE v2.5 16-Feb-2017 18:07:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @cropMovieGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @cropMovieGUI_OutputFcn, ...
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


% --- Executes just before cropMovieGUI is made visible.
function cropMovieGUI_OpeningFcn(hObject,eventdata,handles,varargin)

% Check input
% The mainFig and procID should always be present
% procCOnstr and procName should only be present if the concrete process
% initation is delegated from an abstract class. Else the constructor will
% be directly read from the package constructor list.
ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addOptional('MD',[],@(x)isa(x,'MovieData'));
ip.addParamValue('mainFig',-1,@ishandle);
ip.parse(hObject,eventdata,handles,varargin{:});
userData.MD =ip.Results.MD;
userData.mainFig =ip.Results.mainFig;
        
% Set up copyright statement
set(handles.text_copyright, 'String', getLCCBCopyright());

% Set up available input channels
set(handles.listbox_selectedChannels,'String',userData.MD.getChannelPaths(), ...
    'UserData',1:numel(userData.MD.channels_));

% Save the image directories and names (for cropping preview)
userData.nFrames = userData.MD.nFrames_;
userData.imRectHandle.isvalid=0;
userData.cropROI = [1 1 userData.MD.imSize_(end:-1:1)];
userData.previewFig=-1;

% Read the first image and update the sliders max value and steps
props = get(handles.listbox_selectedChannels, {'UserData','Value'});
userData.chanIndx = props{1}(props{2});
set(handles.edit_frameNumber,'String',1);
set(handles.slider_frameNumber,'Min',1,'Value',1,'Max',userData.nFrames,...
    'SliderStep',[1/max(1,double(userData.nFrames-1))  10/max(1,double(userData.nFrames-1))]);
userData.imIndx=1;
userData.imData=mat2gray(userData.MD.channels_(userData.chanIndx).loadImage(userData.imIndx));
    
set(handles.listbox_selectedChannels,'Callback',@(h,event) update_data(h,event,guidata(h)));
set(handles.edit_firstFrame,'String',1);
set(handles.edit_lastFrame,'String',userData.nFrames);

% Choose default command line output for cropMovieGUI
handles.output = hObject;

% Update user data and GUI data
set(hObject, 'UserData', userData);
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = cropMovieGUI_OutputFcn(~, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(~, ~, handles)
% Delete figure
delete(handles.figure1);

% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, ~, handles)
% Notify the package GUI that the setting panel is closed
userData = get(handles.figure1, 'UserData');

if isfield(userData, 'helpFig') && ishandle(userData.helpFig)
   delete(userData.helpFig) 
end

if ishandle(userData.previewFig), delete(userData.previewFig); end

set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);


% --- Executes on key press with focus on pushbutton_crop and none of its controls.
function pushbutton_crop_KeyPressFcn(~, eventdata, handles)

if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_crop, [], handles);
end

% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(~, eventdata, handles)

if strcmp(eventdata.Key, 'return')
    pushbutton_done_Callback(handles.pushbutton_crop, [], handles);
end

 % --- Executes on button press in checkbox_crop.
function update_data(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');

% Retrieve the channel index
props=get(handles.listbox_selectedChannels,{'UserData','Value'});
chanIndx = props{1}(props{2});
imIndx = get(handles.slider_frameNumber,'Value');

% Load a new image if either the image number or the channel has been changed
if (chanIndx~=userData.chanIndx) ||  (imIndx~=userData.imIndx)
    % Update image flag and dat
    userData.imData=mat2gray(userData.MD.channels_(chanIndx).loadImage(imIndx));
    userData.updateImage=1;
    userData.chanIndx=chanIndx;
    userData.imIndx=imIndx;
        
    % Update roi
    if userData.imRectHandle.isvalid
        userData.cropROI=getPosition(userData.imRectHandle);
    end    
else
    userData.updateImage=0;
end

% In case of crop previewing mode
if get(handles.checkbox_crop,'Value')
    % Create figure if non-existing or closed
    if ~isfield(userData, 'previewFig') || ~ishandle(userData.previewFig)
        userData.previewFig = figure('Name','Select the region to crop',...
            'DeleteFcn',@close_previewFig,'UserData',handles.figure1);
        userData.newFigure = 1;
    else
        figure(userData.previewFig);
        userData.newFigure = 0;
    end
    
    % Retrieve the image object handle
    imHandle =findobj(userData.previewFig,'Type','image');
    if userData.newFigure || userData.updateImage
        if isempty(imHandle)
            imHandle=imshow(userData.imData);
            axis off;
        else
            set(imHandle,'CData',userData.imData);
        end
    end
        
    if userData.imRectHandle.isvalid
        % Update the imrect position
        setPosition(userData.imRectHandle,userData.cropROI)
    else 
        % Create a new imrect object and store the handle
        userData.imRectHandle = imrect(get(imHandle,'Parent'),userData.cropROI);
        fcn = makeConstrainToRectFcn('imrect',get(imHandle,'XData'),get(imHandle,'YData'));
        setPositionConstraintFcn(userData.imRectHandle,fcn);
    end
else
    % Save the roi if applicable
    if userData.imRectHandle.isvalid, 
        userData.cropROI=getPosition(userData.imRectHandle); 
    end
    % Close the figure if applicable
    if ishandle(userData.previewFig), delete(userData.previewFig); end
end
set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);

function close_previewFig(hObject, eventdata)
handles = guidata(get(hObject,'UserData'));
set(handles.checkbox_crop,'Value',0);
update_data(handles.checkbox_crop, eventdata, handles);


% --- Executes on slider movement.
function frameNumberEdition_Callback(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');

% Retrieve the value of the selected image
if strcmp(get(hObject,'Tag'),'edit_frameNumber')
    frameNumber = str2double(get(handles.edit_frameNumber, 'String'));
else
    frameNumber = get(handles.slider_frameNumber, 'Value');
end
frameNumber=round(frameNumber);

% Check the validity of the frame values
if isnan(frameNumber)
    warndlg('Please provide a valid frame value.','Setting Error','modal');
end
frameNumber = min(max(frameNumber,1),userData.nFrames);

% Store value
set(handles.slider_frameNumber,'Value',frameNumber);
set(handles.edit_frameNumber,'String',frameNumber);

% Save data and update graphics
set(handles.figure1, 'UserData', userData);
guidata(hObject, handles);
update_data(hObject,eventdata,handles);


% --- Executes on button press in pushbutton_outputDirectory.
function pushbutton_outputDirectory_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');
pathname = uigetdir(userData.MD.movieDataPath_,'Select output directory');

% Test uigetdir output and store its results
if isequal(pathname,0), return; end
set(handles.edit_outputDirectory,'String',pathname);

% Save data
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);


% --- Executes on button press in pushbutton_addfile.
function pushbutton_addfile_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');
[filename, pathname]=uigetfile({'*.tif;*.TIF;*.stk;*.STK;*.bmp;*.BMP;*.jpg;*.JPG',...
    'Image files (*.tif,*.stk,*.bmp,*.jpg)'},...
    'Select the reference frame',userData.MD.movieDataPath_);

% Test uigetdir output and store its results
if isequal(pathname,0) || isequal(filename,0), return; end
files = get(handles.listbox_additionalFiles,'String');
if any(strcmp([pathname filename],files)),return; end
files{end+1} = [pathname filename];
set(handles.listbox_additionalFiles,'String',files,'Value',numel(files));

% --- Executes on button press in pushbutton_removeFile.
function pushbutton_removeFile_Callback(hObject, eventdata, handles)

props = get(handles.listbox_additionalFiles,{'String','Value'});
if isempty(props{1}), return; end
files= props{1};
files(props{2})=[];
set(handles.listbox_additionalFiles,'String',files,'Value',max(1,props{2}-1));

% --- Executes on button press in pushbutton_crop.
function pushbutton_crop_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');

% Check valid output directory
outputDirectory = get(handles.edit_outputDirectory,'String');
if isempty(outputDirectory),
    errordlg('Please select an output directory','Error','modal');
end

% Read cropROI if crop window is still visible
if userData.imRectHandle.isvalid
    userData.cropROI=getPosition(userData.imRectHandle);
end

% Read cropTOI
firstFrame = str2double(get(handles.edit_firstFrame,'String'));
lastFrame = str2double(get(handles.edit_lastFrame,'String'));
cropTOI=firstFrame:lastFrame;

additionalFiles= get(handles.listbox_additionalFiles,'String');
if isempty(additionalFiles)
    filesArgs={};
else
    filesArgs={'additionalFiles',additionalFiles};
end
    
% Call the crop routine
MD = cropMovie(userData.MD,outputDirectory,'cropROI',userData.cropROI,...
    'cropTOI',cropTOI,filesArgs{:});   

% If new MovieData was created (from movieSelectorGUI)
if userData.mainFig ~=-1, 
    % Retrieve main window userData
    userData_main = get(userData.mainFig, 'UserData');

    % Append new ROI to movie selector panel
    userData_main.MD = horzcat(userData_main.MD, MD);
    set(userData.mainFig, 'UserData', userData_main)
    movieSelectorGUI('refreshDisplay',userData.mainFig,...
        eventdata,guidata(userData.mainFig));
end
% Delete current window
delete(handles.figure1)

function edit_firstFrame_Callback(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');
firstFrame = str2double(get(handles.edit_firstFrame,'String'));
lastFrame = str2double(get(handles.edit_lastFrame,'String'));
if ~ismember(firstFrame,1:userData.nFrames) || firstFrame>lastFrame    
    set(handles.edit_firstFrame,'String',1)
end

function edit_lastFrame_Callback(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');
firstFrame = str2double(get(handles.edit_firstFrame,'String'));
lastFrame = str2double(get(handles.edit_lastFrame,'String'));
if ~ismember(lastFrame,1:userData.nFrames) || firstFrame>lastFrame     
    set(handles.edit_lastFrame,'String',userData.nFrames)
end
