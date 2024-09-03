function packageGUI_OpeningFcn(hObject,eventdata,handles,packageName,varargin)
% Callback called at the opening of packageGUI
%
% packageGUI_OpeningFcn(packageName,MD)   MD: MovieData object
%
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

% Useful tools
%
% User Data:
%
%       userData.MD - array of MovieData object
%       userData.ML - array of MovieList object
%       userData.ImD - array of ImageData object	% added June 2020
%       userData.package - array of package (same length with userData.MD)
%       userData.crtPackage - the package of current MD
%       userData.id - the id of current MD on board
%
%       userData.dependM - dependency matrix
%       userdata.statusM - GUI status matrix
%       userData.optProcID - optional process ID
%       userData.applytoall - array of boolean for batch movie set up
%
%       userData.passIconData - pass icon image data
%       userData.errorIconData - error icon image data
%       userData.warnIconData - warning icon image data
%       userData.questIconData - help icon image data
%       userData.colormap - color map
%
%       userData.setFig - array of handles of (multiple) setting figures (may not exist)
%       userData.resultFig - array of handles of (multiple) result figures (may not exist)
%       userData.packageHelpFig - handle of (single) help figure (may not exist)
%       userData.iconHelpFig - handle of (single) help figures (may not exist)
%       userData.statusFig - handle of (multiple) status figures (may not exist)
%       userData.processHelpFig - handle of (multiple) help figures (may not exist) 
%       
%
% NOTE:
%   
%   userData.statusM - 1 x m stucture array, m is the number of Movie Data 
%                      this user data is used to save the status of movies
%                      when GUI is switching between different movie(s)
%                   
%   	fields: IconType - the type of status icons, 'pass', 'warn', 'error'
%               Msg - the message displayed when clicking status icons
%               Checked - 1 x n logical array, n is the number of processes
%                         used to save value of check box of each process
%               Visited - logical true or false, if the movie has been
%                         loaded to GUI before 

% Sebastien Besson May 2011 (last modified Sep 2011()

% Add a slider bar to the right of the listed processes when there is a
% large number of processes and the height of the packageGUI is over the screensize.
% Updated by Qiongjing (Jenny) Zou, Jun 2019

% Add ImageData compatibility
% Updated by Qiongjing (Jenny) Zou, Jun 2020

% Input check
ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x));
ip.addRequired('handles',@isstruct);
ip.addRequired('packageName',@ischar);
ip.addOptional('MO',[],@(x) isa(x,'MovieObject'));
ip.addParameter('MD',[],@(x) isempty(x) || isa(x,'MovieData'));
% ip.addParameter('menu_parallel',true, @(x) islogical(x) || x==1 || x==0);
% ip.addParameter('menu_debug',true, @(x) islogical(x) || x==1 || x==0);
ip.addParameter('ML',[],@(x) isempty(x) || isa(x,'MovieList'));
ip.addParameter('ImD',[],@(x) isempty(x) || isa(x,'ImageData'));
ip.addParameter('packageConstr','',@(x) isa(x,'function_handle'));
ip.addParameter('packageIndx',{},@iscell);
ip.addParameter('cluster',[],@(x) isempty(x) || isa(x,'parallel.Cluster'));
ip.parse(hObject,eventdata,handles,packageName,varargin{:});

% Read the package name
packageName = ip.Results.packageName;
assert(any(strcmp(superclasses(packageName),'Package')),...
    sprintf('%s is not a valid Package',packageName));
      
handles.output = hObject;
userData = get(handles.figure1,'UserData');
if isempty(userData), userData = struct(); end
userData.packageName = packageName;
userData.MD = ip.Results.MD;
userData.ML = ip.Results.ML;
userData.ImD = ip.Results.ImD;
if(~isempty(ip.Results.cluster))
    uTrackParCluster(ip.Results.cluster);
end

%If package GUI supplied without argument, saves a boolean which will be
%read by packageNameGUI_OutputFcn
if isempty(ip.Results.MO)
    userData.startMovieSelectorGUI=true;
    set(handles.figure1,'UserData',userData);
    guidata(hObject, handles);
    return
end

if isa(ip.Results.MO,'MovieList')
    userData.ML = ip.Results.MO;
    set(handles.pushbutton_status,'Enable','off');
elseif isa(ip.Results.MO,'ImageData')
    userData.ImD=ip.Results.MO;
else
    userData.MD=ip.Results.MO; 
end

% Call package GUI error
set(handles.text_copyright, 'String', getLCCBCopyright());

% Singleton control
try assert(~userData.init)
catch ME
    if strcmpi(ME.identifier,'MATLAB:nonExistentField');
        userData.init=true;
    else
        return
    end
end


% ----------------------------- Load MovieData ----------------------------
nMovies = numel(ip.Results.MO);
if(isempty(ip.Results.packageIndx))
    packageIndx = cell(1, nMovies);

    % I. Before loading MovieData, firstly check if the current package exists
    for i = 1:nMovies
        % Check for existing packages and create them if false
        packageIndx{i} = ip.Results.MO(i).getPackageIndex(packageName,1,true);
    end
else
    % I (alt). If given the package index, use that instead
    packageIndx = ip.Results.packageIndx;
end


for i = find(~cellfun(@isempty, packageIndx))
    userData.package(i) = ip.Results.MO(i).packages_{packageIndx{i}};
end

if any(cellfun(@isempty, packageIndx))
    % Get the adapted constructor
    if ~isempty(ip.Results.packageConstr),
        packageConstr = ip.Results.packageConstr;
    elseif isConcreteClass(userData.packageName)
        packageConstr = str2func(userData.packageName);
    else
        % Launch interface to determine constructor
        concretePackages = eval([userData.packageName '.getConcretePackages(ip.Results.MO)']);
        [selection, status] = listdlg('Name','',...
            'PromptString',{'Select the type of object';'you want to track:'},...
            'ListString', {concretePackages.name},'SelectionMode','single');
        if ~status,
            userData.startMovieSelectorGUI=true;
            set(handles.figure1,'UserData',userData);
            guidata(hObject, handles); 
            return
        end
        packageConstr = concretePackages(selection).packageConstr;
    end
    
    % Add package to movie
    for i = find(cellfun(@isempty, packageIndx))
        ip.Results.MO(i).addPackage(packageConstr(ip.Results.MO(i),...
            ip.Results.MO(i).outputDirectory_));
        userData.package(i) = ip.Results.MO(i).packages_{end};
    end
end

% Run sanity check to check basic dependencies are satisfied
movieExceptions = cell(nMovies, 1);
for i = 1:nMovies
    try
        userData.package(i).sanityCheck(true,'all');
    catch ME
        movieExceptions{i} = MException('lccb:initialization',...
            '%s initialization',...
            userData.package(i).getName);
        movieExceptions{i} = movieExceptions{i}.addCause(ME);
    end
end

if ~all(cellfun(@isempty, movieExceptions))
    generateReport(movieExceptions, userData);
    userData.startMovieSelectorGUI=true;
    set(handles.figure1,'UserData',userData);
    guidata(hObject, handles);
    return
end

% ------------- Check if existing processes can be recycled ---------------
recyclableProc = cell(1, nMovies);
processClassNames = userData.package(1).getProcessClassNames;

% Multiple movies loop
for i = 1:nMovies
    if isempty(packageIndx{i}) && ~isempty(ip.Results.MO(i).processes_)
        recyclableProcIndx = cellfun(@(x) cellfun(@(y)isa(y,x),...
            ip.Results.MO(i).processes_),processClassNames,'UniformOutput',false);
        recyclableProc{i}=ip.Results.MO(i).processes_(any(vertcat(recyclableProcIndx{:}),1));
    end
end

recyclableProcMovie = find(~cellfun(@isempty, recyclableProc));

if ~isempty(recyclableProcMovie)
                      
    % Ask user if to recycle
    msg = ['Record indicates that existing processes are recyclable for %s package:'...
        '\n\nDo you want to load and re-use these steps?'];
    user_response = questdlg(sprintf(msg,userData.package(1).getName),...
        'Recycle Existing Steps','No','Yes','Yes');
    
    if strcmpi(user_response,'Yes')
        for i = recyclableProcMovie           
            recycleProcessGUI(recyclableProc{i}, userData.package(i),'mainFig', handles.figure1)
        end
    end      
end

% Initialize userdata
userData.id = 1;
userData.crtPackage = userData.package(userData.id);
userData.dependM = userData.package(userData.id).getDependencyMatrix;
userData.optProcID =find(sum(userData.dependM==2,1));
nProc = size(userData.dependM, 1);
userData.statusM = repmat( struct('IconType', {cell(1,nProc)}, 'Msg', {cell(1,nProc)}, 'Checked', zeros(1,nProc), 'Visited', false), 1, nMovies);

% -----------------------Load and set up icons----------------------------

% Load icon images from dialogicons.mat
userData = loadLCCBIcons(userData);

% Set figure colormap
supermap(1,:) = get(hObject,'color');
set(hObject,'colormap',supermap);

userData.colormap = supermap;

% Set up package help. 
set(handles.figure1,'CurrentAxes',handles.axes_help);
Img = image(userData.questIconData); 
set(gca, 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'),...
    'visible','off','YDir','reverse');
set(Img,'ButtonDownFcn',@icon_ButtonDownFcn, 'UserData', struct('class', packageName))
% --------------------------Set up processes------------------------------

% List of template process uicontrols to expand
templateTag{1} = 'checkbox';
templateTag{2} = 'axes_icon';
templateTag{3} = 'pushbutton_show';
templateTag{4} = 'pushbutton_set';
templateTag{5} = 'axes_prochelp';
templateTag{6} = 'pushbutton_open';
templateTag{7} = 'processTagLabel';

set(handles.(templateTag{6}),'CData',userData.openIconData);

% templateTag{6} = 'pushbutton_clear'; To be implemented someday?
procTag=templateTag;


figure1Pos = get(handles.figure1,'Position')+(nProc-1)*[0 0 0 40];
screenSize = get(0,'ScreenSize');

if figure1Pos(4) <= screenSize(4)-70
    
    set(handles.figure1,'Position',...
        get(handles.figure1,'Position')+(nProc-1)*[0 0 0 40])
    set(handles.panel_movie,'Position',...
        get(handles.panel_movie,'Position')+(nProc-1)*[0 40 0 0])
    set(handles.panel_proc,'Position',...
        get(handles.panel_proc,'Position')+(nProc-1)*[0 0 0 40])
    set(handles.text_status, 'Position',...
        get(handles.text_status,'Position')+(nProc-1)*[0 40 0 0])
    
    delete(handles.slider1);
    
else
    default_fig1Pos = get(handles.figure1,'Position');
    set(handles.figure1,'Position',...
        [default_fig1Pos(1) default_fig1Pos(2) default_fig1Pos(3) screenSize(4)-100])

    default_panelMoviePos = get(handles.panel_movie,'Position');
    set(handles.panel_movie,'Position',...
        [default_panelMoviePos(1) screenSize(4)-100-default_panelMoviePos(4) ...
        default_panelMoviePos(3) default_panelMoviePos(4)])

    default_panelProcPos = get(handles.panel_proc,'Position');
    sliderMoveSize = default_panelProcPos(4)+(nProc-1)*40 -(screenSize(4)-100)+default_panelMoviePos(4)+66.8;

    set(handles.panel_proc,'Position',...
        [default_panelProcPos(1) default_panelProcPos(2)-sliderMoveSize ...
        default_panelProcPos(3) default_panelProcPos(4)+(nProc-1)*40])

    default_textStatusPos = get(handles.text_status,'Position');
    set(handles.text_status, 'Position',...
        [default_textStatusPos(1) screenSize(4)-100-165.2 ...
        default_textStatusPos(3) default_textStatusPos(4)])

    new_fig1Pos = get(handles.figure1,'Position');
    default_slider1Pos = get(handles.slider1,'Position');
    set(handles.slider1, 'Position', ...
        [default_slider1Pos(1) default_slider1Pos(2) default_slider1Pos(3) ...
        new_fig1Pos(4)-default_panelMoviePos(4)-66.8])
end
 
% Replicate templates ui controls for each process
for i = 1 : nProc
    for j = 1 : length(templateTag)
        procTag{j}=[templateTag{j} '_' num2str(i)];
        handles.(procTag{j}) = copyobj(handles.(templateTag{j}),handles.panel_proc);
        set(handles.(procTag{j}),'Tag',procTag{j},'Position',...
            get(handles.(templateTag{j}),'Position')+(nProc-i)*[0 40 0 0]);
        if ~strcmp(templateTag{j}(1:4), 'axes')
            % Make sure callbacks are copied - copyobj does not copy
            % callbacks starting with R2014b
            set(handles.(procTag{j}),'Callback',...
                get(handles.(templateTag{j}),'Callback'));
        end
    end
  
    % Set name of the process in the corresponding checkbox
    processClassName = userData.crtPackage.getProcessClassNames{i};
    try
        processName = userData.crtPackage.processes_{i}.name_;
    catch err
        processName=eval([processClassName '.getName']);
    end

    checkboxString = [' Step ' num2str(i) ': ' processName];
    set(handles.(procTag{1}),'String',checkboxString)

    if ~isempty(userData.crtPackage.processes_{i}) ...
        && isprop(userData.crtPackage.processes_{i}, 'tag_') && ~isempty(userData.crtPackage.processes_{i}.tag_)
        processTagLabelString = ['[' userData.crtPackage.processes_{i}.tag_ ']'];
    else
        processTagLabelString = '{no tag}';
    end

    set(handles.(procTag{7}),'String',processTagLabelString)
    set(handles.(procTag{7}),'Visible','off')
    
    % Setup help button
    set(handles.figure1,'CurrentAxes',handles.(procTag{5}));
    Img = image(userData.smallquestIconData);
    set(gca, 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'),...
        'visible','off','YDir','reverse');  
    set(Img,'ButtonDownFcn',@icon_ButtonDownFcn,...
        'UserData', struct('class', processClassName))
end
handles.processTagLabels = findall(0,'-regexp','Tag', 'processTagLabel_');

% Remove templates and remove from the handles structure
cellfun(@(x) delete(handles.(x)), templateTag)
handles = rmfield(handles,templateTag);

% Add text boxes for optional processes
optTag = 'text_optional';
for i = userData.optProcID
    procOptTag=[optTag '_' num2str(i)];
    handles.(procOptTag) = copyobj(handles.(optTag),handles.panel_proc);
    set(handles.(procOptTag),'Tag',procOptTag,'Position',...
        get(handles.(optTag),'Position')+(nProc-i)*[0 40 0 0]);
end

% Remove template optional text box
delete(handles.(optTag));
handles = rmfield(handles,optTag);

% --------------------------Create tools menu-----------------------------

if ~isempty(userData.crtPackage.getTools)
    handles.menu_tools = uimenu(handles.figure1,'Label','Tools','Position',2);
    for i=1:length(userData.crtPackage.getTools)
        toolMenuTag=['menu_tools_' num2str(i)];
        handles.(toolMenuTag) = uimenu(handles.menu_tools,...
            'Label',userData.crtPackage.getTools(i).name,...
            'Callback',@(h,event)menu_tools_Callback(h),'Tag',toolMenuTag);
    end
end

% --------------------------Other GUI settings-----------------------------

% set titles
set(handles.figure1, 'Name',['Control Panel - ' userData.crtPackage.getName]);
set(handles.text_packageName,'String',userData.crtPackage.getName);

% Set movie explorer
msg = {};
if isa(ip.Results.MO,'MovieData'), movieType = 'Movie'; elseif isa(ip.Results.MO,'ImageData'), movieType = 'ImageData'; else movieType = 'Movie list'; end
for i = 1: length(ip.Results.MO)
    msg = horzcat(msg, {sprintf('  %s %d of %d', movieType, i, length(ip.Results.MO))});
end
set(handles.popupmenu_movie, 'String', msg, 'Value', userData.id);

% Set option depen
if length(ip.Results.MO) == 1
    set(handles.checkbox_runall, 'Visible', 'off')
    set(handles.pushbutton_left, 'Enable', 'off')
    set(handles.pushbutton_right, 'Enable', 'off')   
    set(handles.checkbox_all, 'Visible', 'off', 'Value', 0)
    userData.applytoall=zeros(nProc,1);
else
    set(handles.checkbox_runall, 'Visible', 'on')
    userData.applytoall=ones(nProc,1);
end


set(handles.pushbutton_run, 'Callback', @(hObject,eventdata)packageGUI_RunFcn(hObject,eventdata,guidata(hObject)));
% Set web links in menu
set(handles.menu_about_gpl,'UserData','http://www.gnu.org/licenses/gpl.html')
set(handles.menu_about_lccb,'UserData','http://www.utsouthwestern.edu/labs/danuser/')
set(handles.menu_about_lccbsoftware,'UserData','http://www.utsouthwestern.edu/labs/danuser/software/')
% 
% if ~ip.Results.menu_parallel 
%     para_menu_handles = findall(0,'-regexp','Tag','menu_parallel$');
%     para_menu_handles.Enable = 'off';
%     userData.para_menu_handles = 'off';
% end
% 
% if ~ip.Results.menu_debug 
%     debug_menu_handles = findall(0,'-regexp','Tag','menu_debug$');
%     debug_menu_handles.Enable = 'off';
%     userData.debug_menu_handles = 'off';
% end

% Update handles structure
set(handles.figure1,'UserData',userData);
guidata(hObject, handles);
set(Img,'ButtonDownFcn',@icon_ButtonDownFcn);


packageGUI_RefreshFcn(handles, 'initialize')
end


% --------------------------------------------------------------------
function menu_tools_Callback(hObject)

handles =guidata(hObject);
userData = get(handles.figure1, 'UserData');
prop=get(hObject,'Tag');
toolID = str2double(prop(length('menu_tools_')+1:end));

toolHandle=userData.crtPackage.getTools(toolID).funHandle;
userData.toolFig(toolID) = toolHandle('mainFig',handles.figure1);

set(handles.figure1, 'UserData', userData);
end