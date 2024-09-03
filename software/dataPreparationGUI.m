function varargout = dataPreparationGUI(varargin)
% DATAPREPARATIONGUI M-file for dataPreparationGUI.fig
%      DATAPREPARATIONGUI, by itself, creates a new DATAPREPARATIONGUI or raises the existing
%      singleton*.
%
%      H = DATAPREPARATIONGUI returns the handle to a new DATAPREPARATIONGUI or the handle to
%      the existing singleton*.
%
%      DATAPREPARATIONGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DATAPREPARATIONGUI.M with the given input arguments.
%
%      DATAPREPARATIONGUI('Property','Value',...) creates a new DATAPREPARATIONGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before dataPreparationGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to dataPreparationGUI_OpeningFcn via varargin.
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

% Edit the above text to modify the response to help dataPreparationGUI

% Last Modified by GUIDE v2.5 10-Apr-2013 12:15:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @dataPreparationGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @dataPreparationGUI_OutputFcn, ...
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


% --- Executes just before dataPreparationGUI is made visible.
function dataPreparationGUI_OpeningFcn(hObject, ~, handles, varargin)
%
% dataPreparationGUI('mainFig', handles.figure1) - call from movieSelector
%
% Useful tools:
%
% User Data:
% 
% userData.channels - array of Channel objects
% userData.mainFig - handle of movie selector GUI
% userData.handles_main - 'handles' of movie selector GUI
%
% userData.setChannelFig - handle of channel set-up figure
% userData.iconHelpFig - handle of help dialog
%
%
%

set(handles.text_copyright, 'String', getLCCBCopyright())

% Choose default command line output forfor i dataPreparationGUI
handles.output = hObject;

userData = get(handles.figure1, 'UserData');
if isempty(userData), userData = struct(); end

% Initialize the userData
userData.projectDir = pwd;
userData.outputDir = [];
userData.rawData =[];
userData.channels =[];
userData.movies=[];

% Load help icon from dialogicons.mat
userData = loadLCCBIcons(userData);
supermap(1,:) = get(hObject,'color');
userData.colormap = supermap;

axes(handles.axes_help);
Img = image(userData.questIconData);
set(hObject,'colormap',supermap);
set(gca, 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'),...
    'visible','off');
set(Img,'ButtonDownFcn',@icon_ButtonDownFcn,...
    'UserData', struct('class',mfilename));

% TestIif the dataPreparationGUI ewas called from the movieSelctorGUI
if nargin > 3       
    t = find(strcmp(varargin,'mainFig'));
    if isempty(t)
        error('User-defined: input error, correct statement: dataPreparationGUI(''mainFig'',handles.figure1)');
    end
    set(handles.checkbox_createMD,'Value',1);
    % Save main figure handles
    userData.mainFig = varargin{t+1};
    userData.handles_main = guidata(userData.mainFig);  
end

% Save data and update graphics
set(handles.figure1,'UserData',userData)
guidata(hObject, handles);
update_graphics(hObject,handles)

% UIWAIT makes dataPreparationGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = dataPreparationGUI_OutputFcn(~, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(~, ~, handles)
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1);

% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(~, ~, handles)
% handles    structure with handles and user data (see GUIDATA)

userData = get(handles.figure1, 'UserData');

% Delete help window if existing
if isfield(userData, 'iconHelpFig') && ishandle(userData.iconHelpFig)
   delete(userData.iconHelpFig) 
end


% --- Executes on button press in pushbutton_delete_channel.
function pushbutton_delete_channel_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_delete_channel (see GCBO)
% handles    structure with handles and user data (see GUIDATA)

userData = get(handles.figure1, 'UserData');

% Retrieve the selected channel id, remove it and update the selected channel
userData.selectedChannel=get(handles.listbox_channels,'Value');
userData.channels(userData.selectedChannel)=[];
userData.selectedChannel = max(userData.selectedChannel-1,1);

% Save data and update graphics
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
update_graphics(hObject,handles)

% --- Executes on button press in pushbutton_edit_channel.
function channelEdition_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_edit_channel (see GCBO)
% handles    structure with handles and user data (see GUIDATA)

userData = get(handles.figure1, 'UserData');

% Retrieve the channel index in the case of channel edition
if strcmp(get(hObject,'Tag'),'pushbutton_edit_channel')
    userData.selectedChannel = get(handles.listbox_channels,'Value');
    if userData.selectedChannel>  numel(userData.channels), return; end
    newChannel=0;
else
    newChannel=1;
end

% Launch a dialog window asking the user for the new channel directory name
propNames = {'excitationWavelength_','emissionWavelength_','exposureTime_'};
prompt = {'Enter the channel as it is encoded in the image names',...
'Enter the channel directory name - optional',...
'Enter the excitation wavelength (in nm) - optional',...
'Enter the emission wavelength (in nm) - optional',...
'Enter the exposure time (in s) - optional'};
num_lines = 1;
if newChannel
    inputData=inputdlg(prompt,'Create new channel',num_lines);
else
    dlg_title = ['Rename channel ' num2str(userData.selectedChannel)];
    defaultValues={userData.channels(userData.selectedChannel).string,...
        userData.channels(userData.selectedChannel).name,...
        num2str(userData.channels(userData.selectedChannel).properties.excitationWavelength_),...
        num2str(userData.channels(userData.selectedChannel).properties.emissionWavelength_),...
        num2str(userData.channels(userData.selectedChannel).properties.exposureTime_)};
    inputData=inputdlg(prompt,dlg_title,num_lines,defaultValues);
end

if isempty(inputData), return; end
% If no channel directory name is supplied, use the channel string name
if isempty(inputData{2}), inputData{2}=inputData{1}; end

% Update the channel string and 
if newChannel
    % Add a new channel structure and update the channel index
    userData.channels(end+1).name=inputData{2};
    userData.channels(end).exportMD = 1;
    userData.channels(end).string='';
    userData.channels(end).properties.excitationWavelength_=[];
    userData.channels(end).properties.emissionWavelength_=[];
    userData.channels(end).properties.exposureTime_=[];
    userData.selectedChannel = numel(userData.channels);
else
    userData.channels(userData.selectedChannel).name=inputData{2};
end
userData.channels(userData.selectedChannel).string=inputData{1};

% Manage input properties
propList=inputData(3:end);
if ~isempty(propList),
    % Check for property values validity
    propValues = cellfun(@str2num,propList','Unif',false);
    validProperties = Channel.checkValue(propNames,propValues);
    
    % Affect valid, non-multiple properties values
    for i= find(validProperties)
        userData.channels(userData.selectedChannel).properties.(propNames{i})=propValues{i};
    end
end

% Save data and update graphics
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
update_graphics(hObject,handles)

% --- Executes on button press in pushbutton_select_projectDir.
function pushbutton_select_projectDir_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_select_projectDir (see GCBO)
% handles    structure with handles and user data (see GUIDATA)

userData = get(handles.figure1, 'UserData');
pathname = uigetdir(userData.projectDir,'Select project folder');

% Test uigetdir output, reinitalize the movies if any, store the project
% directory and the output directory (project directory by default)
if isequal(pathname,0), return; end
userData.projectDir = pathname;
userData.outputDir = pathname;
userData.movies=[];
userData.rawData=[];
set(handles.edit_projectDir,'String',pathname);
set(handles.edit_outputDir,'String',pathname);

% List all subdirectories, including the project directory itself
allSub = dir(userData.projectDir);
allSub = allSub([allSub.isdir]' &  ...
    arrayfun(@(x)(~any(strcmp(x.name,{'..'}))),allSub));
nSub = numel(allSub);

for iSub = 1:nSub
    % List all images of any extension in this directory
    % Use the true returnAll option of imDir
    imageFiles = imDir([userData.projectDir filesep allSub(iSub).name],true);
    
    if ~isempty(imageFiles)
        
        % If some image files are found, there are two behaviours
        % 1- if the directory is the project directory, we will use it as
        % it to create channel folders
        % 2- for any other folder, add a new movie which name is set by the
        % folder name
        if ~strcmp(allSub(iSub).name,'.')
            % Add movie with empty properties
            userData.movies(end+1).name=allSub(iSub).name;
            userData.movies(end).properties.pixelSize_=[];
            userData.movies(end).properties.timeInterval_=[];
            userData.movies(end).properties.numAperture_=[];
            userData.movies(end).properties.camBitdepth_=[];
            rawData.movieIndx = numel(userData.movies);
        else
            rawData.movieIndx=[];
        end
        
        % Get the unique bodyname of the image files
        [~, imageBodyNames,~,imageExt]=cellfun(@getFilenameBody,{imageFiles.name},'Unif',false);        
        [imageBodyNames,uniqueIndx] = unique(imageBodyNames);
        
        for iRawData=1:numel(imageBodyNames)
            rawData.name = imageBodyNames{iRawData};
            rawData.dir = allSub(iSub).name;
            rawData.chanIndx=[];
            rawData.ext=imageExt{uniqueIndx(iRawData)};
            userData.rawData=vertcat(userData.rawData,rawData);
        end
        
    end
end

if isempty(userData.rawData)
    errordlg('No images found in the main project folder or its sub-folders !!');
end

% Save data and update graphics
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
update_graphics(hObject,handles)


% --- Executes on button press in pushbutton_guess.
function pushbutton_guess_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_guess (see GCBO)
% handles    structure with handles and user data (see GUIDATA)
userData = get(handles.figure1, 'UserData');

% Check all raw data components have been affected to a channel
checkChannels = any(cellfun(@isempty,{userData.rawData.chanIndx}));
if checkChannels,
    errordlg('All components have not been affected to a channel','modal');
    return;
end 

% Guess the movie names from the raw data
% Use the part of the string to the left of the channel string
nRawData = numel(userData.rawData);
movieNames=cell(1,nRawData);
for i=1:nRawData
    chanIndx = userData.rawData(i).chanIndx;
    indx=regexp(userData.rawData(i).name,userData.channels(chanIndx).string,'start');
    movieNames{i}=userData.rawData(i).name(1:indx-1);
end

% Find the unique movie names and store the correspondinx index for each
% raw data component
[uniqueMovieNames, ~, movieIndx]= unique(movieNames);
userData.movies = cell2struct(uniqueMovieNames,'name')';
initProperties.pixelSize_=[];
initProperties.timeInterval_=[];
initProperties.numAperture_=[];
initProperties.camBitdepth_=[];
[userData.movies(:).properties]=deal(initProperties);
for i=1:nRawData
    userData.rawData(i).movieIndx=movieIndx(i);
end

% Save data and update graphics
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
update_graphics(hObject,handles)



% --- Executes on button press in checkbox_createMD.
function checkbox_createMD_Callback(hObject, ~, handles)

update_graphics(hObject,handles)

% --- Executes on button press in pushbutton_edit_movie.
function pushbutton_edit_movie_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_edit_movie (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

userData = get(handles.figure1, 'UserData');

if isempty(userData), userData = struct(); end
% Retrieve the selected movie ID and launch a dialog box asking the user
% for the new movie name
selectedMovie = get(handles.listbox_movies,'Value');
propNames = {'pixelSize_','timeInterval_','numAperture_','camBitdepth_'};

% Input dialog common properties
dlg_title = 'Movie edition';
num_lines = 1;
prompt = {...
    'Enter the movie directory name:',...
    'Enter the pixel size (in nm) - optional',...
    'Enter the time interval (in s) - optional',...
    'Enter the numerical aperture - optional',...
    'Enter the camera bit depth - optional',...
    };

% Test for single or multiple selection
if numel(selectedMovie)==1
    defaultValues={...
        userData.movies(selectedMovie).name,...
        num2str(userData.movies(selectedMovie).properties.pixelSize_),...
        num2str(userData.movies(selectedMovie).properties.timeInterval_),...
        num2str(userData.movies(selectedMovie).properties.numAperture_),...
        num2str(userData.movies(selectedMovie).properties.camBitdepth_)};
    inputData=inputdlg(prompt,dlg_title,num_lines,defaultValues);
        
    % Update the movie name and store the list of properties in propList
    userData.movies(selectedMovie).name=inputData{1};
    propList=inputData(2:end);
else
    
    % Initialize default values
    nProps = numel(propNames);
    defaultValues=cellfun(@num2str,cell(1,nProps),'Unif',false);
    for i= 1:nProps
        allPropValues=arrayfun(@(x) x.properties.(propNames{i}),userData.movies(selectedMovie),...
            'UniformOutput',false);
        uniquePropValues = unique([allPropValues{:}]);
        emptyPropValues = find(cellfun(@isempty,allPropValues),1);
        unicityTest = isempty(uniquePropValues) ||...
            (numel(uniquePropValues)==1 && isempty(emptyPropValues));
        if unicityTest
            defaultValues{i} = num2str(uniquePropValues);
        else
            defaultValues{i}='(multiple)';
        end
    end
    % Remove the movie name from the prompt list and store answer in
    % propList
    propList=inputdlg(prompt(2:end),dlg_title,num_lines,defaultValues);
    
end

if isempty(propList), return; end
% Check multiple properties
nonMultipleOutput=~strcmp(propList','(multiple)');
% Check for property values validity
propValues = cellfun(@str2num,propList','Unif',false);
validProperties = MovieData.checkValue(propNames,propValues);

% Affect valid, non-multiple properties values 
for i= find(nonMultipleOutput & validProperties)
    for j=selectedMovie
        userData.movies(j).properties.(propNames{i})=propValues{i};
    end
end

% Save data and update graphics
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
update_graphics(hObject,handles)


% --- Executes on selection change in listbox_channels.
function listbox_channels_Callback(hObject, ~, handles)
% hObject    handle to listbox_channels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Check if the user double clicked on the list box
if strcmp(get(handles.figure1,'SelectionType'),'open')
    userData = get(handles.figure1, 'UserData');
    
    % Modify the export status of the selected channel
    userData.selectedChannel=get(handles.listbox_channels,'Value');
    userData.channels(userData.selectedChannel).exportMD=~userData.channels(userData.selectedChannel).exportMD;
    
    % Save data and update graphics
    set(handles.figure1,'UserData',userData);
    guidata(hObject, handles);
    update_graphics(hObject,handles)
end

function update_graphics(hObject, handles)
userData = get(handles.figure1, 'UserData');

if ~isempty(userData.channels)
    % Generate the channel list, put the channels to be exported in the
    % MovieData in bold (using html block), update the channel listbox
    channelList =arrayfun(@(x)[x.name ' (' x.string ')'],userData.channels,'UniformOutput',false);
    if get(handles.checkbox_createMD,'Value')
        exportMDChannels = logical([userData.channels.exportMD]);
        channelList(exportMDChannels)=cellfun(@(x) ['<html><b>' x '</b></html>'],channelList(exportMDChannels),...
            'UniformOutput',false);
    end
    set(handles.listbox_channels,'String',channelList,'Value',userData.selectedChannel);
else
    set(handles.listbox_channels,'String',[]);
end

if ~isempty(userData.movies)
    % Generate the movie list,  update the movies listbox
    set(handles.listbox_movies,'String',{userData.movies.name});
else
    set(handles.listbox_movies,'String',[]);
end

if isempty(userData.rawData), 
    set(handles.listbox_rawData,'String',[]);
    return;
end
nRawData = numel(userData.rawData);
% If raw data has been selected, find the match between each channel string
% with all the raw data components
nChannels = numel(userData.channels(:));
channelMatches=zeros(nChannels,nRawData);
for i=1:nChannels
    findChannel = @(x) (~isempty(regexp(x,userData.channels(i).string,'ONCE')));
    channelMatches(i,:) =  cellfun(findChannel,{userData.rawData.name});    
end

% Test the unique correspondance between each raw data component and a
% channel
for i=1:nRawData
    chanIndx=find(channelMatches(:,i));
    if numel(chanIndx)>1
        errorMsg=sprintf('Multiple user-defined channels match the movie:\n %s',userData.rawData(i).name);
        errordlg(errorMsg);
        userData.rawData(i).chanIndx=[];
        break
    else
        userData.rawData(i).chanIndx = chanIndx;
    end
end

% Update the number of found chanels and put the matched raw data components
% associated with a channel in bold font.
channelMatch = ~cellfun(@isempty,{userData.rawData.chanIndx});
channelMatchText=['Matching channels: ' num2str(sum(channelMatch)) '/' num2str(nRawData)];
set(handles.text_channelMatch,'String',channelMatchText);

rawDataNames={userData.rawData.name};
rawDataNames(channelMatch)=cellfun(@(x) ['<html><b>' x '</b></html>'],rawDataNames(channelMatch),'UniformOutput',false);
set(handles.listbox_rawData,'String',rawDataNames);

% Update the number of found movies
movieMatch = ~cellfun(@isempty,{userData.rawData.movieIndx});
movieMatchText=['Matching movies: ' num2str(sum(movieMatch)) '/' num2str(nRawData)];
set(handles.text_movieMatch,'String',movieMatchText);

% Save data
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);


% --- Executes on button press in pushbutton_select_outputDir.
function pushbutton_select_outputDir_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_select_outputDir (see GCBO)
% handles    structure with handles and user data (see GUIDATA)

userData = get(handles.figure1, 'UserData');
pathname = uigetdir('Select output directory', userData.projectDir);

% Test uigetdir output and store its results
if isequal(pathname,0), return; end
userData.outputDir = pathname;
set(handles.edit_outputDir,'String',pathname);

% Save data
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);


% --- Executes on button press in pushbutton_exclude_component.
function pushbutton_exclude_component_Callback(hObject, ~, handles)
% hObject    handle to pushbutton_exclude_component (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

userData = get(handles.figure1, 'UserData');

% Retrieve the selected component id,
selectedComponents=get(handles.listbox_rawData,'Value');

% Find any movie associated with component to exclude
uniqueMovieIndx =unique([userData.rawData(selectedComponents).movieIndx]);
if ~isempty(uniqueMovieIndx),
    % Loop backwards on found movies (for index renumbering)
    for i=uniqueMovieIndx(end:-1:1)
        % Find movies to remove, i.e. movies where all components are to be
        % excluded        
        nMovieComponentsToExclude = sum([userData.rawData(selectedComponents).movieIndx]==i);
        nMovieComponentsTotal = sum([userData.rawData(:).movieIndx]==i);
        if nMovieComponentsToExclude~=nMovieComponentsTotal, continue; end

        % Remove movie from the list
        userData.movies(i)=[];
        % Renumber movie index of remaining components
        renumberIndx=([userData.rawData(:).movieIndx]>i);
        % Lame for loop as I didn't find a way to avoid it
        for rawDataIndx =find(renumberIndx)
            userData.rawData(rawDataIndx).movieIndx=userData.rawData(rawDataIndx).movieIndx -1;
        end
    end
end

% Remove component and update the selected value
userData.rawData(selectedComponents)=[];
selectedComponents = max(min(selectedComponents)-1,1);
set(handles.listbox_rawData,'Value',selectedComponents);

% Save data and update graphics
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
update_graphics(hObject,handles)


% --- Executes on button press in pushbutton_done.
function pushbutton_done_Callback(~, eventdata, handles)

userData = get(handles.figure1, 'UserData');

% Check all movies and channels have been attributed
if isempty(userData.rawData), return; end
    
checkMovies = any(cellfun(@isempty,{userData.rawData.movieIndx}));
if checkMovies,
    errordlg('All components have not been affected to a movie','modal');
    return;
end

checkChannels = any(cellfun(@isempty,{userData.rawData.chanIndx}));
if checkChannels,
    errordlg('All components have not been affected to a channel','modal');
    return;
end 

wtBar = waitbar(0,'Please wait, ...'); 

% Read checkbox values
createMD=get(handles.checkbox_createMD,'Value');
splitSTK=get(handles.checkbox_splitSTK,'Value');

% Retrieve main window userData if applicable
if isfield(userData,'mainFig')
    userData_main = get(userData.mainFig, 'UserData');
    contentlist = get(userData.handles_main.listbox_movie, 'String');
end

nMovies=numel(userData.movies);
for iMovie=1:nMovies
    % Create the movie folder
    movieFolder=[userData.outputDir filesep userData.movies(iMovie).name];
    if ~isdir(movieFolder), mkdir(movieFolder); end
    
    waitbar(iMovie/nMovies,wtBar,['Please wait, setting up movie ' num2str(iMovie) ' ...']);
    
    % Initialize Channel array (to be fetched in the MovieData constructor)
    if createMD, MDChannels=[]; end
    % Find all raw data components associated with that movie
    rawDataList=find([userData.rawData.movieIndx]==iMovie);
    for nRawData=rawDataList
        % Retrieve the channel index and create the channel folder
        chanIndx = userData.rawData(nRawData).chanIndx;
        channelFolder = [movieFolder filesep userData.channels(chanIndx).name];
        if ~isdir(channelFolder), mkdir(channelFolder); end
        
        % Retrieve original rawDatafolder and copy/move the file
        rawDataFolder = [userData.projectDir filesep userData.rawData(nRawData).dir];
        rawDataBodyName = userData.rawData(nRawData).name;
        rawDataExt = userData.rawData(nRawData).ext;
        rawDataFiles=[rawDataFolder filesep rawDataBodyName '*' rawDataExt];
        
        if strcmpi(rawDataExt,'.stk') && splitSTK
            % Split STK files
            stkFiles=dir(rawDataFiles);
            if numel(stkFiles)>1
                errordlg('Found more than one STK file for the movie');
                continue;
            end
            try
                currIm = stackRead([rawDataFolder filesep stkFiles(1).name]);
            catch errMess
                disp(['stackRead.m failed: ' errMess.message ' Trying tif3Dread.m instead...'])
                %Tif3Dread is slower, but can open some files which the
                %current stackRead version fails to open
                currIm = tif3Dread([rawDataFolder filesep stkFiles(1).name]);
            end
            nIm = size(currIm,3); %Check number of images
            %Get number of digits for writing file names
            nDig = floor(log10(nIm)+1);
            %Make the string for formatting
            fString = strcat('%0',num2str(nDig),'.f');
            %Write them all to new dir
            disp(['Splitting "' stkFiles(1).name '" into ' num2str(nIm) ' seperate files...'])
            for k = 1:nIm
                imwrite(squeeze(currIm(:,:,k)),[channelFolder filesep stkFiles(1).name(1:end-4) '_' num2str(k,fString) '.tif']);
            end
            
        else            
            if get(handles.checkbox_copy,'Value')
                copyfile(rawDataFiles,channelFolder);
            else
                movefile(rawDataFiles,channelFolder);
            end
        end

        % Create a Channel object, and append it to the existing Channel
        % array if applicable
        if createMD && userData.channels(chanIndx).exportMD
            try
                newChannel = Channel(channelFolder);
                set(newChannel, userData.channels(chanIndx).properties);
            catch ME
                throw(ME);
            end
            MDChannels = horzcat(MDChannels, newChannel);
        end
        
    end

    if createMD
        % Check all required channels have been set up
        nFoundChannels =numel(MDChannels);
        nExpectedChannels = sum([userData.channels.exportMD]);
        if nFoundChannels ~= nExpectedChannels
            errordlg('Not all channels were found. Skipped Movie Data creation!');
            continue
        end
        
        % Save Movie Data to disk
        movieDataFile = 'movieData.mat';
        % Launch the MovieData constructor
        try
            MD = MovieData(MDChannels, movieFolder);

        catch ME
            errormsg = sprintf([ME.message '.\n\nCreating movie data failed.']);
            errordlg(errormsg, 'User Input Error','modal');
            continue;
        end

        % Check the movieData sanity
        try
            MD.movieDataPath_=movieFolder;
            MD.movieDataFileName_=movieDataFile;
            set(MD,userData.movies(iMovie).properties);
            MD.sanityCheck;
        catch ME
            delete(MD);
            errormsg = sprintf('%s.\n\nPlease check your movie data. Movie data is not saved.',ME.message);
            errordlg(errormsg,'Channel Error','modal');
            continue;
        end
        
        % Save the movie data
        MD.save;
        
        % Update main window components and controls if applicable
        if isfield(userData,'mainFig')
            if any(strcmp([movieFolder filesep movieDataFile], contentlist))
                errordlg('Cannot overwrite a movie data file which is already in the movie list. Please choose another file name or another path.','Error','modal');
                return
            end

            % Append new ROI to movie selector panel
            userData_main.MD = horzcat(userData_main.MD, MD);
            set(userData.mainFig, 'UserData', userData_main)
            movieSelectorGUI('refreshDisplay',userData.mainFig,...
                eventdata,guidata(userData.mainFig));
        end
    end
end

close(wtBar)

% Update main window components and controls if applicable
if isfield(userData,'mainFig')% Save the data
    set(userData.mainFig, 'UserData', userData_main)
end
% Delete current window
delete(handles.figure1)
