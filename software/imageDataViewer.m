function mainFig = imageDataViewer(MO,varargin)
%MOVIEVIEWER creates a graphical interface to display the analysis output of a MovieObject
% 
% h = imageDataViewer(ImD)
%%%%%%%%% h = imageDataViewer(ImL, 'movieIndex', 3);
%
% This function reads the components of a MovieObject including all
% drawable anlaysis results (determined by the getDrawableOutput method).
% It then generates a graphical interface allowing to switch between image
% results and toggle on/off overlay components. Additionally, two
% interfaces can be created: one to control movie display options (for the
% image and the various overlays) and one interface showing the different
% graph results (i.e. results displayed on separate figures).
% 
% Input 
%
%   MO - the MovieObject to be displayed. If a ImageList is input, the main
%   interface will have a popupmenu allowing to switch between the list and
%   all the movie components.
% 
%   procId - Optional. An array containing the indices of the processes 
%   which output should be displayed by default. Default: empty.
%
%   Optional parameters in param/value pairs
%
%   movieIndex - Integer. For movie list input. If 0 display the movie list
%   and its analysis. If non-zero, set the index of the movie to be
%   displayed. Default: 0.
%
%   showProcTag - logical: displays process tags along with Process names
%                          for disambiguation.
%
% Output:
%   
%   mainFig - the handle of the main control interface
%
% See also: graphViewer, movieViewerOptions
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

% Adapted from movieViewer
% Changed necessary 'MovieData' to 'ImageData', channels_ to imFolders_, nFrames_ to max(cell2mat(userData.MO.reader.nImages)).
% by Qiongjing (Jenny) Zou, Jun 2020

% Check input
ip = inputParser;
ip.addRequired('MO',@(x) isa(x,'MovieObject'));
ip.addOptional('procId',[],@isnumeric);
ip.addOptional('refresher',[],@isstr);
ip.addParameter('movieIndex',0,@isscalar);
ip.addParameter('showProcTag',1,@islogical);
ip.parse(MO,varargin{:});

if strcmp(ip.Results.refresher, '1') == 1
    movieviewerRefresher(MO);
    return
end



% Generate the main figure
mainFig=figure('Name','Viewer','Position',[0 0 200 200],...
    'NumberTitle','off','Tag','figure1','Toolbar','none','MenuBar','none',...
    'Color',get(0,'defaultUicontrolBackgroundColor'),'Resize','off',...
    'DeleteFcn', @(h,event) deleteViewer());
userData=get(mainFig,'UserData');
set(mainFig, 'UserData', userData);

% Read the MovieObject and process index input 
if isa(ip.Results.MO,'ImageList')
    userData.ML=ip.Results.MO;
    userData.movieIndex=ip.Results.movieIndex;
    if userData.movieIndex~=0
        userData.MO=ip.Results.MO.getMovies{userData.movieIndex};
    else
        userData.MO=ip.Results.MO;
    end
        
    userData.procId = ip.Results.procId;
    if ~isempty(ip.Results.procId)
        procId = userData.MO.getProcessIndex(class(userData.ML.processes_{ip.Results.procId}));
    else
        procId = ip.Results.procId;
    end
else
    userData.MO=ip.Results.MO;
    procId=ip.Results.procId;
end
% Check existence of viewer interface
if isfield('MO', 'isMock') && ~MO.isMock() % QZ 'MO' is not a struc, should not use isfield?
    h=findobj(0,'Name','Viewer');
    h2 = findobj('Name', 'hcsview');
    if ~isempty(h), delete(h); end
    if ~isempty(h2), delete(h2); end
else
    hPosition = 10;
end

% Read all drawable output
validProcId= find(cellfun(@(x) ismember('getDrawableOutput',methods(x)) &&...
    x.success_,userData.MO.processes_));
validProc=userData.MO.processes_(validProcId);

% Classify movieData processes by type (image, overlay or graph)
getOutputType = @(type) cellfun(@(x) any(~cellfun(@isempty,...
    regexp({x.getDrawableOutput.type},type,'once','start'))), validProc);
isImageProc = getOutputType('image');
imageProc=validProc(isImageProc);
imageProcId = validProcId(isImageProc);
isOverlayProc = getOutputType('[oO]verlay');
overlayProc = validProc(isOverlayProc);
overlayProcId = validProcId(isOverlayProc);
isGraphProc =getOutputType('[gG]raph');
graphProc=validProc(isGraphProc);
graphProcId = validProcId(isGraphProc);

% Create series of anonymous function to generate process controls
createProcText= @(panel,i,j,pos,hgt,name) uicontrol(panel,'Style','text',...
    'Position',[10 pos 250 hgt],'Tag',['text_process' num2str(i)],...
    'String',name,'HorizontalAlignment','left','FontWeight','bold');
createProcTextTag= @(panel,i,j,pos,name) uicontrol(panel,'Style','text',...
    'Position',[10 pos 250 10],'Tag',['text_processTag' num2str(i)],...
    'String',name,'HorizontalAlignment','left','FontWeight','normal','FontSize', 7,'ForegroundColor',[0 .45 .74]);
createOutputText= @(panel,i,j,pos,text) uicontrol(panel,'Style','text',...
    'Position',[40 pos 200 20],'Tag',['text_process' num2str(i) '_output'...
    num2str(j)],'String',text,'HorizontalAlignment','left');
createProcButton= @(panel,i,j,k,pos) uicontrol(panel,'Style','radio',...
    'Position',[200+30*k pos 20 20],'Tag',['radiobutton_process' num2str(i) '_output'...
    num2str(j) '_channel' num2str(k)]);
createChannelBox= @(panel,i,j,k,pos,varargin) uicontrol(panel,'Style','checkbox',...
    'Position',[200+30*k pos 20 20],'Tag',['checkbox_process' num2str(i) '_output'...
    num2str(j) '_channel' num2str(k)],varargin{:});
createMovieBox= @(panel,i,j,pos,name,varargin) uicontrol(panel,'Style','checkbox',...
    'Position',[40 pos 200 25],'Tag',['checkbox_process' num2str(i) '_output'...
    num2str(j)],'String',[' ' name],varargin{:});

%% Image panel creation % QZ good now, check out checkChannelOutput
if isa(userData.MO,'ImageData')
    imagePanel = uibuttongroup(mainFig,'Position',[0 0 1/2 1],...
        'Title','Image','BackgroundColor',get(0,'defaultUicontrolBackgroundColor'),...
        'Units','pixels','Tag','uipanel_image');
        
    % Create controls for switching between process image output
    hPosition=10;
    nProc = numel(imageProc);
    for iProc=nProc:-1:1
        output=imageProc{iProc}.getDrawableOutput;
        validChan = imageProc{iProc}.checkChannelOutput;
        validOutput = find(strcmp({output.type},'image'));
        for iOutput=validOutput(end:-1:1)
            createOutputText(imagePanel,imageProcId(iProc),iOutput,hPosition,output(iOutput).name);
            if strfind(output(iOutput).var,'merged')
                createProcButton(imagePanel,imageProcId(iProc),iOutput,1,hPosition);
            else
                arrayfun(@(x) createProcButton(imagePanel,imageProcId(iProc),iOutput,x,hPosition),...
                    find(validChan));            
            end
            hPosition=hPosition+20;
        end
        if size(imageProc{iProc}.name_,2)>32 % make sure long process name fit on GUI
            hgt = 40;
        else
            hgt = 20;
        end  
        createProcText(imagePanel,imageProcId(iProc),iOutput,hPosition,hgt,imageProc{iProc}.name_);
        hPosition=hPosition+hgt;
        
        % Show process tags for disambiguation
        if ip.Results.showProcTag
            createProcTextTag(imagePanel,imageProcId(iProc),iOutput,hPosition,imageProc{iProc}.tag_);
            hPosition=hPosition+15;
        end

    end
    
    % Create controls for selecting channels (raw image)
    hPosition=hPosition+10;
    uicontrol(imagePanel,'Style','radio','Position',[10 hPosition 200 20],...
        'Tag','radiobutton_channels','String',' Raw images','Value',1,...
        'HorizontalAlignment','left','FontWeight','bold');
    arrayfun(@(i) uicontrol(imagePanel,'Style','checkbox',...
        'Position',[200+30*i hPosition 20 20],...
        'Tag',['checkbox_channel' num2str(i)],'Value',0,...
        'Callback',@(h,event) redrawChannel(h,guidata(h))),...
        1:numel(userData.MO.imFolders_));
    
    hPosition=hPosition+20;
    uicontrol(imagePanel,'Style','text','Position',[120 hPosition 100 20],...
        'Tag','text_channels','String','ImFolders');
    arrayfun(@(i) uicontrol(imagePanel,'Style','text',...
        'Position',[200+30*i hPosition 20 20],...
        'Tag',['text_channel' num2str(i)],'String',i),...
        1:numel(userData.MO.imFolders_));    
else
    imagePanel=-1;
end

%% Overlay panel creation % QZ I do not need this
if ~isempty(overlayProc)
    overlayPanel = uipanel(mainFig,'Position',[1/2 0 1/2 1],...
        'Title','Overlay','BackgroundColor',get(0,'defaultUicontrolBackgroundColor'),...
        'Units','pixels','Tag','uipanel_overlay');
    
    % Create overlay options
    hPosition=10;
    nProc = numel(overlayProc);
    for iProc=nProc:-1:1
        output=overlayProc{iProc}.getDrawableOutput;
        
        % Create checkboxes for movie overlays
        validOutput = find(strcmp({output.type},'movieOverlay'));
        for iOutput=validOutput(end:-1:1)
            createMovieBox(overlayPanel,overlayProcId(iProc),iOutput,hPosition,output(iOutput).name,...
                'Callback',@(h,event) redrawOverlay(h,guidata(h)));
            hPosition=hPosition+20;
        end
        
        % Create checkboxes for channel-specific overlays
        validOutput = find(strcmp({output.type},'overlay'));
        for iOutput=validOutput(end:-1:1)
            validChan = overlayProc{iProc}.checkChannelOutput;
            createOutputText(overlayPanel,overlayProcId(iProc),iOutput,hPosition,output(iOutput).name);
            arrayfun(@(x) createChannelBox(overlayPanel,overlayProcId(iProc),iOutput,x,hPosition,...
                'Callback',@(h,event) redrawOverlay(h,guidata(h))),find(validChan));
            hPosition=hPosition+20;
        end
        if size(overlayProc{iProc}.name_,2)>32  % make sure long process name fit on GUI
            hgt = 40;
        else
            hgt = 20;
        end
        createProcText(overlayPanel,overlayProcId(iProc),iOutput,hPosition,hgt,overlayProc{iProc}.name_);
        hPosition=hPosition+hgt;
    end
    
    if ~isempty(overlayProc)
        uicontrol(overlayPanel,'Style','text','Position',[120 hPosition 100 20],...
            'Tag','text_channels','String','Channels');
        arrayfun(@(i) uicontrol(overlayPanel,'Style','text',...
            'Position',[200+30*i hPosition 20 20],...
            'Tag',['text_channel' num2str(i)],'String',i),...
            1:numel(userData.MO.channels_));
    end
else
    overlayPanel=-1;
end

%% Get image/overlay panel size and resize them % QZ good now
imagePanelSize = getPanelSize(imagePanel);
overlayPanelSize = getPanelSize(overlayPanel);
panelsLength = max(500,imagePanelSize(1)+overlayPanelSize(1));
panelsHeight = max([imagePanelSize(2),overlayPanelSize(2)]);

% Resize panel
if ishandle(imagePanel)
    set(imagePanel,'Position',[10 panelsHeight-imagePanelSize(2)+10 ...
        imagePanelSize(1) imagePanelSize(2)],...
        'SelectionChangeFcn',@(h,event) redrawImage(guidata(h)))
end
if ishandle(overlayPanel)
    set(overlayPanel,'Position',[imagePanelSize(1)+10 panelsHeight-overlayPanelSize(2)+10 ...
        overlayPanelSize(1) overlayPanelSize(2)]);
end


%% Create movie panel % QZ good now
moviePanel = uipanel(mainFig,...
    'Title','','BackgroundColor',get(0,'defaultUicontrolBackgroundColor'),...
    'Units','pixels','Tag','uipanel_movie','BorderType','none');


    
if isa(userData.MO,'ImageData')
    hPosition=10;
 
    % Create controls for scrollling through the movie if regular moviedata
    MO = userData.MO;
    if isa(MO,'ImageData') && ~MO.isHCS()
    uicontrol(moviePanel, 'Style', 'togglebutton','String', 'Run images',...
        'Position', [10 hPosition 100 20],'Callback',@(h,event) runMovie(h,guidata(h)));

% QZ comment out below movie exporting features
%     % Create control button for exporting figures and movie (cf Francois' GUI)
% 
%     uicontrol(moviePanel, 'Style', 'checkbox','Tag','checkbox_saveFrames',...
%         'Value',0,'String', 'Save frames','Position', [150 hPosition 100 20]);
%     uicontrol(moviePanel, 'Style', 'checkbox','Tag','checkbox_saveMovie',...
%         'Value',0,'String', 'Save movie','Position', [250 hPosition 100 20]);
%     uicontrol(moviePanel, 'Style', 'popupmenu','Tag','popupmenu_movieFormat',...
%         'Value',1,'String', {'MOV';'AVI'},'Position', [350 hPosition 100 20]);    
        
        
    hPosition = hPosition+30;
    uicontrol(moviePanel,'Style','text','Position',[10 hPosition 50 15],...
        'String','NO.','Tag','text_frame','HorizontalAlignment','left');
    uicontrol(moviePanel,'Style','edit','Position',[70 hPosition 30 20],...
        'String','1','Tag','edit_frame','BackgroundColor','white',...
        'HorizontalAlignment','left',...
        'Callback',@(h,event) redrawScene(h,guidata(h)));
    uicontrol(moviePanel,'Style','text','Position',[100 hPosition 40 15],...
        'HorizontalAlignment','left',...
        'String',['/' num2str(max(cell2mat(userData.MO.reader.nImages)))],'Tag','text_frameMax');
    
    uicontrol(moviePanel,'Style','slider',...
        'Position',[150 hPosition panelsLength-160 20],...
        'Value',1,'Min',1,'Max',max(cell2mat(userData.MO.reader.nImages)),...
        'SliderStep',[1/double(max(cell2mat(userData.MO.reader.nImages)))  5/double(max(cell2mat(userData.MO.reader.nImages)))],...
        'Tag','slider_frame','BackgroundColor','white',...
        'Callback',@(h,event) redrawScene(h,guidata(h)));
    
    
% QZ comment out 3D slider for imageDataViewer
%     %%%% 3D slider starts %%%%
%     if MO.is3D() 
%         hPosition = hPosition+30;
%         uicontrol(moviePanel, 'Style', 'togglebutton','String', 'Show in 3D',...
%             'Position', [10 hPosition 100 20],'Callback',@(h,event) render3DMovie(h,guidata(h)));
%        
%         uicontrol(moviePanel, 'Style', 'checkbox','Tag','checkbox_saveplane',...
%             'Value',0,'String', 'Save plane','Position', [150 hPosition 100 20]);
%         uicontrol(moviePanel, 'Style', 'checkbox','Tag','checkbox_saveRender',...
%             'Value',0,'String', 'Save render','Position', [250 hPosition 100 20]);
%         uicontrol(moviePanel, 'Style', 'popupmenu','Tag','popupmenu_3DmovieFormat',...
%             'Value',1,'String', {'MOV';'AVI'},'Position', [350 hPosition 100 20]);
%         
%         
%         hPosition = hPosition+30;
%         uicontrol(moviePanel,'Style','text','Position',[10 hPosition 50 15],...
%             'String','Depth','Tag','text_depth','HorizontalAlignment','left');
%         uicontrol(moviePanel,'Style','edit','Position',[70 hPosition 30 20],...
%             'String','1','Tag','edit_depth','BackgroundColor','white',...
%             'HorizontalAlignment','left',...
%             'Callback',@(h,event) redrawScene(h,guidata(h)));
%         uicontrol(moviePanel,'Style','text','Position',[100 hPosition 40 15],...
%             'HorizontalAlignment','left',...
%             'String',['/' num2str(userData.MO.zSize_)],'Tag','text_depthMax');
%         
%         uicontrol(moviePanel,'Style','slider',...
%             'Position',[150 hPosition panelsLength-160 20],...
%             'Value',1,'Min',1,'Max',userData.MO.zSize_,...
%             'SliderStep',[1/double(userData.MO.zSize_)  5/double(userData.MO.zSize_)],...
%             'Tag','slider_depth','BackgroundColor','white',...
%             'Callback',@(h,event) redrawScene(h,guidata(h)));
% %         userData.projectionAxis3D = 'Z'; % defined XY, ZX, or ZY  (Z,Y,X)
%     end
%     
%     %%% 3D slider ends %%%%
    
    hPosition = hPosition+30;
    elseif isa(MO,'ImageData') && MO.isMock()
%         shtext = uicontrol(moviePanel, 'Style', 'text', 'Position', [10 60 panelsLength-100 20], ...
%             'String', ['Preview for site ', MO.channels_(1,1).getGenericName(MO.channels_(1,1).hcsPlatestack_{1}, 'site_on')], ...
%         'HorizontalAlignment','left');
    if size(MO.mockMD_.index,1) > 1
    %hPosition = hPosition+20;
    uicontrol(moviePanel,'Style','text','Position',[10 hPosition 50 15],...
        'String','Frame','Tag','text_frame','HorizontalAlignment','left');
    uicontrol(moviePanel,'Style','text','Position',[70 hPosition 80 15],...
        'String','1','Tag','edit_frame','BackgroundColor','white',...
        'HorizontalAlignment','left',...
        'Callback',@(h,event) redrawScene(h,guidata(h)));
    uicontrol(moviePanel,'Style','text','Position',[150 hPosition 40 15],...
        'HorizontalAlignment','left',...
        'String',['/' num2str(userData.MO.nFrames_)],'Tag','text_frameMax');
    
    uicontrol(moviePanel,'Style','slider',...
        'Position',[200 hPosition panelsLength-160 20],...
        'Value',1,'Min',1,'Max',userData.MO.nFrames_,...
        'SliderStep',[1/double(userData.MO.nFrames_)  5/double(userData.MO.nFrames_)],...
        'Tag','slider_frame','BackgroundColor','white',...
        'Callback',@(h,event) redrawScene(h,guidata(h)));    
    end
    hPosition = hPosition+30;
    else
        hcshid = uicontrol('Parent', mainFig, 'Visible', 'off', 'Tag','slider_frame');
        hcsview = figure('Name','hcsview','Position',[100 100 200 200],...
            'NumberTitle','off','Tag','hcsview','Toolbar','none','MenuBar','none',...
            'Color',get(0,'defaultUicontrolBackgroundColor'),'Resize','off',...
            'DeleteFcn', @(h,event) deleteViewer());
        hcsuserData.lastclick = '0';
        hcsuserData.MO = MO;
        set(hcsview, 'UserData', hcsuserData);
        dcolor = get(0, 'DefaultUIControlBackgroundColor');
        platestack = MO.channels_(1,1).hcsPlatestack_;
        wellflag = MO.channels_(1,1).hcsFlags_.wellF;
        % gui for a plate
        %         hp = uipanel('Parent', moviePanel,'Title','Plate Inspector','FontSize',10,...
%             'Position',[10 320 22*((size(wellflag,1)+2)) 22*((size(wellflag,2)+2))]);
        hp = uipanel('Parent', hcsview,'Title','Plate Inspector','FontSize',10,...
             'Position',[.04 .29 .9 .68]);
        % gui for well
        hpp = uipanel('Parent', hcsview,'Title','Well Inspector','FontSize',10,...
            'Position',[.04 .04 .4 .24]); 
        pwgc = 0; % plate-well good count 
        for i2 = 1:size(wellflag,1)
            for i3 = 1:size(wellflag,2)
                
                if i2 == 1
                    hth = uicontrol('Parent', hp, 'Style', 'Text', 'Position', [i3*22+22,370, 22,22],...
                        'String', num2str(i3));
                end
                if i3 == 1
                    htv = uicontrol('Parent', hp, 'Style', 'Text', 'Position', [22,365-i2*22, 22,22],...
                        'String', char(i2-1+'A'));
                end
                if wellflag(i2, i3) == 0
                hb = uicontrol('Parent', hp,'Style','pushbutton', 'Position', [i3*22+22,370-i2*22,22,22]...
                    ,'Tag','slider_frame',...
                    'enable', 'off'); %
                elseif wellflag(i2, i3) == 1
                    pwgc = pwgc + 1;
                    if pwgc == 1
                        ipr1 = i2; ipr2 = i3;
                    end
                    if ~isempty(platestack{i2-ipr1+1, i3-ipr2+1}{1,1})
                    wellname = MO.channels_(1,1).getGenericName(platestack{i2-ipr1+1, i3-ipr2+1}{1,1});
                    else
                        wellname = '';
                    end
                    hb = uicontrol('Parent', hp,'Style','pushbutton', 'Position', [i3*22+22,370-i2*22,22,22]...
                        ,'TooltipString',[wellname, ...
                        ',',num2str(length(platestack{i2-ipr1+1, i3-ipr2+1})), ...
                        ' Sites'],'Tag', wellname,...
                        'UserData', dcolor, ...
                        'enable', 'on',...
                        'Callback', {@updatewellp, platestack, [i2-ipr1+1, i3-ipr2+1], hpp, mainFig, MO});
                    % setting the colors while building the plate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if MO.hasMock()
                        mMDsite_number = [];
                        for imMDt = 1:size(MO.mMDparent_,1)
                            if size(MO.mMDparent_{imMDt},1) > 1 % multi-frame mockMD
                                for imMD = 1:size(MO.mMDparent_{imMDt},1)
                                    if i2-ipr1+1 == MO.mMDparent_{imMDt}(imMD, 1) && i3-ipr2+1 == MO.mMDparent_{imMDt}(imMD,2)
                                        set(hb,'BackgroundColor','cyan');
                                        set(hb,'UserData',[0 1 1]);
                                        mMDsite_number = [mMDsite_number; MO.mMDparent_{imMDt}(imMD,3)];
                                        set(hb, 'Callback', {@updatewellp, platestack, {i2-ipr1+1, i3-ipr2+1, mMDsite_number, 1, MO.mMDparent_{imMDt,2}}, hpp, mainFig, MO});
                                    end
                                end
                                
                            else
                                if i2-ipr1+1 == MO.mMDparent_{imMDt}(1) && i3-ipr2+1 == MO.mMDparent_{imMDt}(2)
                                    set(hb,'BackgroundColor','green');
                                    set(hb,'UserData',[0 1 0]);
                                    mMDsite_number = [mMDsite_number; MO.mMDparent_{imMDt}(3)];
                                    set(hb, 'Callback', {@updatewellp, platestack, {i2-ipr1+1, i3-ipr2+1, mMDsite_number, 0, MO.mMDparent_{imMDt,2}}, hpp, mainFig, MO});
                                end
                            end
                        end
                    end
                    
                end
                
            end
        end
        uicontrol('Parent', hpp, 'Style', 'pushbutton', 'Position', [299 55 180 25], ...
            'String', 'Preprocess Controls ', ...
        'Callback', {@togglecontrol, MO});
        
        if ~isempty(MO.mMDparent_)
        uicontrol('Parent', hpp, 'Style', 'pushbutton', 'Position', [299 30 180 25], ...
            'String', 'Flush Controls', ...
            'Callback', {@flushcontrol, MO});
        end
        handles = guihandles(hcsview);
        guidata(handles.hcsview, handles);
        set(handles.hcsview,'UserData',hcsuserData);
    end
end
% Create movie location edit box
uicontrol(moviePanel,'Style','text','Position',[3 hPosition 60 20],...
    'String','ImgData','Tag','text_movie');

% Create popupmenu if input is a ImageList, else  list the movie path
if isa(ip.Results.MO,'ImageList')
    moviePaths = cellfun(@getDisplayPath,userData.ML.getMovies,'UniformOutput',false);
    movieIndex=0:numel(moviePaths);
    
    uicontrol(moviePanel,'Style','popupmenu','Position',[60 hPosition panelsLength-110 20],...
        'String',vertcat(getDisplayPath(ip.Results.MO),moviePaths(:)),'UserData',movieIndex,...
        'Value',find(userData.movieIndex==movieIndex),...
        'HorizontalAlignment','left','BackgroundColor','white','Tag','popup_movie',...
        'Callback',@(h,event) switchMovie(h,guidata(h)));
    if userData.movieIndex==0, set(findobj(moviePanel,'Tag','text_movie'),'String','List'); end
    
else
    uicontrol(moviePanel,'Style','edit','Position',[60 hPosition panelsLength-110 20],...
        'String',getDisplayPath(ip.Results.MO),...
        'HorizontalAlignment','left','BackgroundColor','white','Tag','edit_movie');
end

% Add help button
set(0,'CurrentFigure',mainFig)
hAxes = axes('Units','pixels','Position',[panelsLength-50 hPosition  20 20],...
    'Tag','axes_help', 'Parent', moviePanel);
icons = loadLCCBIcons();
Img = image(icons.questIconData);
set(hAxes, 'XLim',get(Img,'XData'),'YLim',get(Img,'YData'), 'visible','off','YDir','reverse');
set(Img,'ButtonDownFcn',@icon_ButtonDownFcn, 'UserData', struct('class','movieViewer'));

% Add copyright
hPosition = hPosition+30;
uicontrol(moviePanel,'Style','text','Position',[10 hPosition panelsLength-100 20],...
    'String',getLCCBCopyright(),'Tag','text_copyright',...
    'HorizontalAlignment','left');

% Get overlay panel size
moviePanelSize = getPanelSize(moviePanel);
moviePanelHeight =moviePanelSize(2);
if isa(MO, 'MovieData') % QZ I do not need this
    if ~isempty(MO.channels_(1,1).hcsFlags_)
        if max(size(MO.channels_(1,1).hcsFlags_.wellF)) ~= 0
            hcsl = 22*size(MO.channels_(1,1).hcsFlags_.wellF,2)+150;
            hcsh = 22*size(MO.channels_(1,1).hcsFlags_.wellF,1)+250;
        else
            hcsl = 0;
            hcsh = 0;
        end
        panelsLength = max(panelsLength, hcsl);
    end
end

    %% Resize panels and figure % QZ good now
    sz=get(0,'ScreenSize');
    maxWidth = panelsLength+20;
    maxHeight = panelsHeight+moviePanelHeight;
    if isa(MO, 'ImageData')
        if MO.isMock()
            maxHeight = 190;
        end
        if MO.isHCS() && ~MO.isMock()
            set(hcsview,'Position',[sz(3)/50 sz(4)/2 maxWidth hcsh]);
        end
        
    end
set(mainFig,'Position',[sz(3)/50 sz(4)/3.8 maxWidth maxHeight]);
set(moviePanel,'Position',[10 panelsHeight+10 panelsLength moviePanelHeight]);
% Update handles structure and attach it to the main figure
handles = guihandles(mainFig);
guidata(handles.figure1, handles);


% Create redraw callbacks
userData.redrawImageFcn = @(varargin) redrawImage(handles, varargin{:});
userData.redrawOverlaysFcn = @(varargin) redrawOverlays(handles, varargin{:});
userData.getFigure = @(figName) getFigure(handles, figName);
set(handles.figure1,'UserData',userData);

% Create options figure % QZ I do not need the movieViewerOptions window now.
if isa(userData.MO, 'MovieData')
    optionsFig = movieViewerOptions(mainFig);
    set(optionsFig, 'Tag', 'optionsFig');
end
%% Add additional panel for independent graphs % QZ I do not need this
if ~isempty(graphProc)
    graphFig = graphViewer(mainFig, graphProc, graphProcId, intersect(graphProcId,procId));
    set(graphFig, 'Tag', 'graphFig');
end
%% Set up default parameters
% Auto check input process
for i=intersect(procId,validProcId)
    h=findobj(mainFig,'-regexp','Tag',['(\w)_process' num2str(i)  '_output1.*'],...
        '-not','Style','text');
    set(h,'Value',1);
end

% QZ comment out below specific feature for PointSourceDetectionProcess3DDynROI and TrackingDynROIProcess
% % if procId was assigned and PointSourceDetectionProcess3DDynROI or TrackingDynROIProcess was selected on Overlay panel,
% % then selected radiobutton on Image panel should be changed from Raw image to BuildDynROIProcess.
% if ~isempty(i) && ~isempty(h) && isequal(h.Style, 'checkbox') && isequal(imagePanel.SelectedObject.String, ' Raw image')
%     switch class(validProc{i})
%         case {'PointSourceDetectionProcess3DDynROI', 'TrackingDynROIProcess'}
%             BuildDynROIProcId = validProcId(cellfun(@(x) isa(x, 'BuildDynROIProcess'),validProc));
%             if ~isempty(BuildDynROIProcId)
%                 h2 = findobj(mainFig,'-regexp','Tag',['radiobutton_process' num2str(BuildDynROIProcId)  '_output2.*']);
%                 set(h2,'Value',1);
%             else
%                 set(h,'Value',0);
%             end
%     end
% end

% Clear cache when initializing movieViewer
cached.load('-clear');

if isa(MO, 'ImageData')
    if userData.MO.isMock()
        redrawScene(handles.figure1, handles);
    end
end

% Update the image and overlays
if isa(userData.MO,'ImageData') && ~userData.MO.isHCS()
    redrawScene(handles.figure1, handles);
    addMovieViewerKeyboardShortcuts(mainFig, getFigure(handles,'Image'));
end

%%%%%%%%% function imageDataViewer ends here %%%%%%%%%


% function slider_callback(src,eventdata,panel)
% pos=get(panel,'Position');
% pos(2)=(1-pos(4))*get(src,'Value');
% set(panel,'Position',pos)
% uistack(panel,'top');


function displayPath= getDisplayPath(movie)
[~,endPath] = fileparts(movie.getPath);
displayPath = fullfile(endPath,movie.getFilename);

function switchMovie(hObject,handles)
userData=get(handles.figure1,'UserData');
props=get(hObject,{'UserData','Value'});
if isequal(props{1}(props{2}), userData.movieIndex),return;end
if isempty(userData.procId)
   movieViewer(userData.ML,'movieIndex',props{1}(props{2})); 
else
movieViewer(userData.ML,'procId', userData.procId,'movieIndex',props{1}(props{2}));
end

function size = getPanelSize(hPanel)
if ~ishandle(hPanel), size=[0 0]; return; end
a=get(get(hPanel,'Children'),'Position');
P=vertcat(a{:});
size = [max(P(:,1)+P(:,3))+10 max(P(:,2)+P(:,4))+20];

function runMovie(hObject,handles)

userData = get(handles.figure1, 'UserData');
nFrames = max(cell2mat(userData.MO.reader.nImages));
startFrame = get(handles.slider_frame,'Value');
if startFrame == nFrames, startFrame =1; end;
if get(hObject,'Value')
    action = 'Stop'; 
else
    action = 'Run';
end
set(hObject,'String',[action ' images']);

% QZ comment out movie exporting features:
% % Get frame/movies export status
% saveMovie = get(handles.checkbox_saveMovie,'Value');
% saveFrames = get(handles.checkbox_saveFrames,'Value');
% props = get(handles.popupmenu_movieFormat,{'String','Value'});
% movieFormat = props{1}{props{2}};
% 
% if saveMovie
%     moviePath = fullfile(userData.MO.outputDirectory_,['Movie.' lower(movieFormat)]);
% end
% 
% % Initialize movie output
% if saveMovie && strcmpi(movieFormat,'mov')
%     MakeQTMovie('start',moviePath);
%     MakeQTMovie('quality',.9)
% end
% 
% if saveMovie && strcmpi(movieFormat,'avi')
%     movieFrames(1:nFrames) = struct('cdata', [],'colormap', []);
% end
% 
% % Initialize frame output
% if saveFrames;
%     fmt = ['%0' num2str(ceil(log10(nFrames))) 'd'];
%     frameName = @(frame) ['frame' num2str(frame, fmt) '.tif'];
%     fpath = [userData.MO.outputDirectory_ filesep 'Frames'];
%     mkClrDir(fpath);
%     fprintf('Generating movie frames:     ');
% end

for iFrame = startFrame : nFrames
    if ~get(hObject,'Value'), return; end % Handle pushbutton press
    set(handles.slider_frame, 'Value',iFrame);
    redrawScene(hObject, handles);    
    drawnow;
    
    % Get current frame for frame/movie export
    hFig = getFigure(handles,'Image');
% QZ comment out movie exporting features:
%     if saveMovie && strcmpi(movieFormat,'mov'), MakeQTMovie('addfigure'); end
%     if saveMovie && strcmpi(movieFormat,'avi'), movieFrames(iFrame) = getframe(hFig); end
%     if saveFrames
%         print(hFig, '-dtiff', fullfile(fpath,frameName(iFrame)));
%         fprintf('\b\b\b\b%3d%%', round(100*iFrame/(nFrames)));
%     end
end

% QZ comment out movie exporting features:
% % Finish frame/movie creation
% if saveFrames; fprintf('\n'); end
% if saveMovie && strcmpi(movieFormat,'mov'), MakeQTMovie('finish'); end
% 
% if saveMovie && strcmpi(movieFormat,'avi') 
%     v = VideoWriter(moviePath);
%     open(v);
%     writeVideo(v, movieFrames);
%     close(v); 
% end

% Reset button
set(hObject,'String', 'Run images', 'Value', 0);

function render3DMovie(hObject,handles)
userData = get(handles.figure1, 'UserData');

% Update userData if BuildDynROIProcess was selected on Image panel.
imageTag = get(get(handles.uipanel_image,'SelectedObject'),'Tag');
userData = isBuildDynROIProcessSelected(imageTag, userData);

nPlane = userData.MO.zSize_;

startFrame = get(handles.slider_depth,'Value');
if startFrame == nPlane, startFrame =1; end;
if get(hObject,'Value')
   action = 'Stop'; 
else action = 'Run'; 
end
set(hObject,'String',[action ' Rendering']);

% Get frame/movies export status
saveMovie = get(handles.checkbox_saveRender,'Value');
saveFrames = get(handles.checkbox_saveplane,'Value');
props = get(handles.popupmenu_3DmovieFormat,{'String','Value'});
movieFormat = props{1}{props{2}};

if saveMovie,
    moviePath = fullfile(userData.MO.outputDirectory_,['Movie.' lower(movieFormat)]);
end

% Initialize movie output
if saveMovie && strcmpi(movieFormat,'mov')
    MakeQTMovie('start',moviePath);
    MakeQTMovie('quality',.9)
end

if saveMovie && strcmpi(movieFormat,'avi')
    movieFrames(1:nPlane) = struct('cdata', [],'colormap', []);
end

% Initialize frame output
if saveFrames;
    fmt = ['%0' num2str(ceil(log10(nPlane))) 'd'];
    frameName = @(frame) ['depth' num2str(frame, fmt) '.tif'];
    fpath = [userData.MO.outputDirectory_ filesep 'Depths'];
    mkClrDir(fpath);
    fprintf('Generating movie depth:     ');
end

imageTagBegin = get(get(handles.uipanel_image,'SelectedObject'),'Tag');

for iFrame = startFrame : nPlane
    imageTagNow = get(get(handles.uipanel_image,'SelectedObject'),'Tag');
    if ~isequal(imageTagBegin, imageTagNow)
        % If selected object is changed on the Image panel while the 3D rendering is running,
        % stop the 3D rendering!
        set(hObject,'String', 'Show 3D', 'Value', 0);
        return;
    else
        
        if ~get(hObject,'Value'), return; end % Handle pushbutton press
        set(handles.slider_depth, 'Value',iFrame);
        redrawScene(hObject, handles);
        drawnow;
        
        % Get current frame for frame/movie export
        hFig = getFigure(handles,'Image');
        if saveMovie && strcmpi(movieFormat,'mov'), MakeQTMovie('addfigure'); end
        if saveMovie && strcmpi(movieFormat,'avi'), movieFrames(iFrame) = getframe(hFig); end
        if saveFrames
            print(hFig, '-dtiff', fullfile(fpath,frameName(iFrame)));
            fprintf('\b\b\b\b%3d%%', round(100*iFrame/(nPlane)));
        end
    end
end

% Finish frame/movie creation
if saveFrames; fprintf('\n'); end
if saveMovie && strcmpi(movieFormat,'mov'), MakeQTMovie('finish'); end
if saveMovie && strcmpi(movieFormat,'avi')
    v = VideoWriter(moviePath);
    open(v);
    writeVideo(v, movieFrames);
    close(v); 
end

% Reset button
set(hObject,'String', 'Show 3D', 'Value', 0);
        

function redrawScene(hObject, handles)
userData = get(handles.figure1, 'UserData');
% Retrieve the value of the selected image
if userData.MO.isHCS()
    if userData.MO.isMock()
        if size(userData.MO.mockMD_.index,1) == 1;
        frameNumber = 1;
        else
        frameNumber = get(handles.slider_frame,'Value');
        end    
    else
        frameNumber = get(handles.slider_frame,'Value');
        if max(size(frameNumber)) ~= 1
            frameNumber = frameNumber{1};
            if frameNumber == 0;
                frameNumber = 1;
            end
        end
    end
elseif strcmp(get(hObject,'Tag'),'edit_frame')
    frameNumber = str2double(get(handles.edit_frame, 'String'));
else
    frameNumber = get(handles.slider_frame, 'Value');
end
frameNumber=round(frameNumber);
frameNumber = min(max(frameNumber,1),max(cell2mat(userData.MO.reader.nImages)));

% QZ comment out 3D depth aquisition for imageDataViewer:
% %3D depth aquisition
% if userData.MO.is3D() && strcmp(get(hObject,'Tag'),'edit_depth')
%     ZNr = str2double(get(handles.edit_depth,'String'));
% elseif userData.MO.is3D()
%     ZNr = round(get(handles.slider_depth,'Value'));
% end
    
% Set the slider and editboxes values
if ~userData.MO.isHCS()
    set(handles.edit_frame,'String',frameNumber);
    set(handles.slider_frame,'Value',frameNumber);
    if userData.MO.is3D()
        set(handles.edit_depth,'String',ZNr);
        set(handles.slider_depth,'Value',ZNr);
    end
end
if userData.MO.isMock() && size(userData.MO.mockMD_.index,1) > 1
set(handles.slider_frame,'Value',frameNumber);
set(handles.edit_frame,'String',userData.MO.channels_(1,1).getGenericName(userData.MO.channels_(1,1).hcsPlatestack_{frameNumber}, 'site_on'));
if userData.MO.isMock() && size(userData.MO.mockMD_.index,1) == 1
    set(handles.slider_frame,'Value',1);
end
end

% Update the image and overlays
redrawImage(handles);
redrawOverlays(handles);

function h= getFigure(handles,figName)
    
userData = get(handles.figure1,'UserData');
    
if(~isfield(userData,'figures'))
    userData.figures = struct();
end
   
if(isfield(userData.figures,figName) ...
        && ishandle(userData.figures.(figName)) ...
        && isvalid(handle(userData.figures.(figName))))
    h = userData.figures.(figName);
else
    h = findobj(0,'-regexp','Name',['^' figName '$']);
end
if ~isempty(h)
    figure(h);
    try
        userData.figures.(figName) = h;
        set(handles.figure1,'UserData',userData);
    catch err
        switch(err.identifier)
            case 'MATLAB:AddField:InvalidFieldName'
                % figName may not be a proper field name
                % Ignore error
            otherwise
                rethrow(err)
        end
    end
    return;
end

%Create a figure
if strcmp(figName,'Image')
    
    sz=get(0,'ScreenSize');
    nx=max(cell2mat(userData.MO.reader.sizeXmax)); % nx=userData.MO.imSize_(2);
    ny=max(cell2mat(userData.MO.reader.sizeYmax)); % ny=userData.MO.imSize_(1);
    sc = max(1, max(nx/(.9*sz(3)), ny/(.9*sz(4))));
    h = figure('Position',[sz(3)*.2 sz(4)*.2 nx/sc ny/sc],...
        'Name',figName,'NumberTitle','off','Tag','viewerFig',...
        'UserData',handles.figure1);
    
    % figure options for movie export
    iptsetpref('ImShowBorder','tight');
    set(h, 'InvertHardcopy', 'off');
    set(h, 'PaperUnits', 'Points');
    set(h, 'PaperSize', [nx ny]);
    set(h, 'PaperPosition', [0 0 nx ny]); % very important
    set(h, 'PaperPositionMode', 'auto');
    % set(h,'DefaultLineLineSmoothing','on');
    % set(h,'DefaultPatchLineSmoothing','on');
    
    axes('Parent',h,'XLim',[0 nx],...
        'YLim',[0 ny],'Position',[0.05 0.05 .9 .9]);
    userData.figures.Movie = h;
    set(handles.figure1,'UserData',userData);
    
    % Set the zoom properties
    hZoom=zoom(h);
    hPan=pan(h);
    set(hZoom,'ActionPostCallback',@(h,event)panZoomCallback(h));
    set(hPan,'ActionPostCallback',@(h,event)panZoomCallback(h));
    
    addMovieViewerKeyboardShortcuts(0,h);
else
    h = figure('Name',figName,'NumberTitle','off','Tag','viewerFig');
end


function redrawChannel(hObject,handles)

% Callback for channels checkboxes to avoid 0 or more than 4 channels
channelBoxes = findobj(handles.figure1,'-regexp','Tag','checkbox_channel*');
nChan=numel(find(arrayfun(@(x)get(x,'Value'),channelBoxes)));
if nChan==0, set(hObject,'Value',1); elseif nChan>3, set(hObject,'Value',0); end

redrawImage(handles)

function redrawImage(handles,varargin)

imageTag = get(get(handles.uipanel_image,'SelectedObject'),'Tag');
% Get the figure handle
drawFig = getFigure(handles,'Image');
userData=get(handles.figure1,'UserData');
if userData.MO.isMock() && size(userData.MO.mockMD_.index,1) == 1
    frameNr = 1;
else
    frameNr = get(handles.slider_frame,'Value');
end
if userData.MO.is3D()
    ZNr = get(handles.slider_depth, 'Value');
else
    ZNr = 1;
end

% Use corresponding method depending if input is channel or process output
channelBoxes = findobj(handles.figure1,'-regexp','Tag','checkbox_channel*');
[~,index]=sort(arrayfun(@(x) get(x,'Tag'),channelBoxes,'UniformOutput',false));
channelBoxes =channelBoxes(index);
if strcmp(imageTag,'radiobutton_channels')
    set(channelBoxes,'Enable','on');
    chanList=find(arrayfun(@(x)get(x,'Value'),channelBoxes));
    if isempty(chanList)
        set(channelBoxes(1), 'Value', 1);
        chanList=find(arrayfun(@(x)get(x,'Value'),channelBoxes));
    end
    userData.MO.imFolders_(chanList).draw(frameNr,ZNr,varargin{:});
    displayMethod = userData.MO.imFolders_(chanList(1)).displayMethod_;
    projectionAxis3D = 'Z'; % Just default
    procId = 0;
    if userData.MO.is3D()
        if get(handles.slider_depth, 'Max') ~= userData.MO.zSize_
            set(handles.text_depthMax, 'String',['/' num2str(userData.MO.zSize_)]);
            set(handles.slider_depth, 'Max',userData.MO.zSize_,...
                'SliderStep',[1/double(userData.MO.zSize_)  5/double(userData.MO.zSize_)]);
            set(handles.edit_depth,'String',ZNr);
        end
    end
else
    set(channelBoxes,'Enable','off');
    % Retrieve the id, process nr and channel nr of the selected imageProc
    tokens = regexp(imageTag,'radiobutton_process(\d+)_output(\d+)_channel(\d+)','tokens');
    procId=str2double(tokens{1}{1});
    outputList = userData.MO.processes_{procId}.getDrawableOutput;
    iOutput = str2double(tokens{1}{2});
    output = outputList(iOutput).var;
    iChan = str2double(tokens{1}{3});
    if userData.MO.is3D
% QZ comment out below part specific for BuildDynROIProcess
%         if isa(userData.MO.processes_{procId}, 'BuildDynROIProcess')
%             s = cached.load(userData.MO.processes_{procId}.outFilePaths_{3, iChan}, '-useCache', true);
%             load(s.movieDataDynROICell{1});
%             newMD = MD; % MD built based on DynROI raw images.
%             clear MD;
%             if get(handles.slider_depth, 'Max') ~= newMD.zSize_ || get(handles.slider_depth, 'Value') > newMD.zSize_
%                 if ZNr > newMD.zSize_
%                     ZNr = newMD.zSize_;
%                     set(handles.edit_depth,'String',ZNr);
%                     set(handles.slider_depth,'Value',ZNr);
%                 end
%                 set(handles.text_depthMax, 'String',['/' num2str(newMD.zSize_)]);
%                 set(handles.slider_depth, 'Max',newMD.zSize_,...
%                     'SliderStep',[1/double(newMD.zSize_)  5/double(newMD.zSize_)]);
%             end
%         end
        
        userData.MO.processes_{procId}.draw(iChan,frameNr, ZNr, 'output',output, varargin{:});
        % Check for projected Axis in output name
        if strfind(output, 'three')
            projectionAxis3D = 'three';
        elseif ~isempty(strfind(output,'Z')) && ~isempty(strfind(output, 'Y'))
            projectionAxis3D = 'X';
        elseif ~isempty(strfind(output,'Z')) && ~isempty(strfind(output, 'X'))
            projectionAxis3D = 'Y';
        else
            projectionAxis3D = 'Z';
        end
    else
        userData.MO.processes_{procId}.draw(iChan,frameNr, 'output',output,varargin{:});
        projectionAxis3D = 'Z';
    end
    displayMethod = userData.MO.processes_{procId}.displayMethod_{iOutput,iChan};
end

% Disable unrelated overlay processes based on the seletection on Image
% panel for 'NewUTrack3DPackage'.
% Made sure it still work even when packages_ was not added, or when only
% part of the processes were run.
if isfield(handles, 'uipanel_overlay')
    overlayProcTexts = findobj(handles.uipanel_overlay,'-regexp','Tag','^text_process\d$');
    if ~isempty(overlayProcTexts)
        overlayProcTextsTags = arrayfun(@(x) get(x,'Tag'),overlayProcTexts, 'UniformOutput',false);
        matches = regexp(overlayProcTextsTags,'\d*','Match');
        overlayProcIds = str2double([matches{:}]);
        overlayProcNames =  cell(1,max(overlayProcIds));
        for i = overlayProcIds
            overlayProcNames{i} = class(userData.MO.processes_{i});
        end
        
        if procId > 0
            switch class(userData.MO.processes_{procId})
                case {'RenderFullMIPProcess', 'BuildDynROIProcess', 'RenderDynROIMIPProcess'}
                    unrelatedProcNames = userData.MO.processes_{procId}.unrelatedProc();
                    tokens = regexp(overlayProcNames, strjoin(unrelatedProcNames,'|'),'tokens');
                    index = ~cellfun(@isempty,tokens);
                otherwise
                    index = false(size(overlayProcNames));
            end
        else
            % PointSourceDetectionProcess3DDynROI and TrackingDynROIProcess are
            % unrelated processes for raw image
            tokens = regexp(overlayProcNames, strjoin({'^PointSourceDetectionProcess3DDynROI$', '^TrackingDynROIProcess$'},'|'),'tokens');
            index = ~cellfun(@isempty,tokens);
        end
        
        overlayTextsAndBoxes = findobj(handles.uipanel_overlay,'-regexp','Tag','\w*_process*');
        set(overlayTextsAndBoxes,'Enable','on');
        if sum(index) > 0
            overlayTagsAll = arrayfun(@(x) get(x,'Tag'),overlayTextsAndBoxes, 'UniformOutput',false);
            newIndex = false(size(overlayTagsAll,1),1);
            newIndexCheckbox = false(size(overlayTagsAll,1),1);
            for j = find(index)
                tokens = regexp(overlayTagsAll,['\w*_process' num2str(j)],'tokens');
                newIndex = ~cellfun(@isempty,tokens)+newIndex;
                tokensCheckbox = regexp(overlayTagsAll,['checkbox_process' num2str(j)],'tokens');
                newIndexCheckbox = ~cellfun(@isempty,tokensCheckbox)+newIndexCheckbox;
            end
            set(overlayTextsAndBoxes(find(newIndex)),'Enable','off');
            % Make sure the unrelated processes' checkboxes get unchecked,
            % and delete what have been drawn on the overlay by those
            % unrelated processes' checkboxes.
            overlayUnrelatedCheckboxes = overlayTextsAndBoxes(find(newIndexCheckbox));
            unrelatedCheckboxObj = findobj(overlayUnrelatedCheckboxes,'Value', 1);
            if ~isempty(unrelatedCheckboxObj)
                set(unrelatedCheckboxObj,'Value', 0);
                for i = 1:numel(unrelatedCheckboxObj)
                    redrawOverlay(unrelatedCheckboxObj(i),handles);
                end
            end
        end
    end
end


optFig = findobj(0,'-regexp','Name','Movie options');
if ~isempty(optFig), 
    userData = get(optFig,'userData');
    userData.setImageOptions(drawFig, displayMethod);
    if userData.MO.is3D
        userData.setProjectionAxis3D(projectionAxis3D)
    end
end
redrawOverlays(handles);
function panZoomCallback(varargin) 
    % Find if options figure exist
optionsFig = findobj(0,'-regexp','Tag', 'optionsFig');
if ~isempty(optionsFig)
    % Reset the scaleBar
    handles = guidata(optionsFig);
    scalebarCallback = get(handles.edit_imageScaleBar,'Callback');
    timeStampCallback = get(handles.checkbox_timeStamp,'Callback');
    scalebarCallback(optionsFig);
    timeStampCallback(optionsFig);
end

function redrawOverlays(handles)
if ~isfield(handles,'uipanel_overlay'), return; end

overlayBoxes = findobj(handles.uipanel_overlay,'-regexp','Tag','checkbox_process*');
checkedBoxes = logical(arrayfun(@(x) get(x,'Value'),overlayBoxes));
overlayTags=arrayfun(@(x) get(x,'Tag'),overlayBoxes(checkedBoxes),...
    'UniformOutput',false);
for i=1:numel(overlayTags),
    redrawOverlay(handles.(overlayTags{i}),handles)
end

function redrawOverlay(hObject,handles) %%%% need configuration for 3D.
    userData=get(handles.figure1,'UserData');
    if userData.MO.isMock()
        if size(userData.MO.mockMD_.index,1) == 1;
            frameNr = 1;
        else
            frameNr = get(handles.slider_frame,'Value');
        end
    else
        frameNr = get(handles.slider_frame,'Value');
    end
if max(size(frameNr)) > 1
frameNr = frameNr{1};
end
overlayTag = get(hObject,'Tag');

% Get figure handle or recreate figure
movieFig = findobj(0,'Name','Movie');
if isempty(movieFig),  redrawScene(hObject, handles); return; end
figure(movieFig);
% Retrieve the id, process nr and channel nr of the selected imageProc
tokens = regexp(overlayTag,'^checkbox_process(\d+)_output(\d+)','tokens');
procId=str2double(tokens{1}{1});
outputList = userData.MO.processes_{procId}.getDrawableOutput;
iOutput = str2double(tokens{1}{2});
output = outputList(iOutput).var;

% Discriminate between channel-specific processes annd movie processes
tokens = regexp(overlayTag,'_channel(\d+)$','tokens');
if ~isempty(tokens)
    iChan = str2double(tokens{1}{1});
    inputArgs={iChan,frameNr};
    graphicTag =['process' num2str(procId) '_channel'...
        num2str(iChan) '_output' num2str(iOutput)];
    movieOverlay = false;
else
    iChan = [];
    inputArgs={frameNr};
    graphicTag = ['process' num2str(procId) '_output' num2str(iOutput)];
    movieOverlay = true;
end
% Get options figure handle
optFig = findobj(0,'-regexp','Name','Movie options');
if ~isempty(optFig), userData = get(optFig, 'userData'); end

% Draw or delete the overlay depending on the checkbox value
if get(hObject,'Value')
    if ~isempty(optFig),
        options = userData.getOverlayOptions();
    else
        options = {};
    end    
    if userData.MO.is3D() % && userData.MO.processes_{procId}.is3DP()
        ZNr = get(handles.slider_depth,'Value');
        userData.MO.processes_{procId}.draw(inputArgs{:},'output',output,... % draw method of process object modificiation for 3D!!!
        options{:},'movieOverlay', movieOverlay,'iZ', ZNr);%, 'projectionAxis3D', userData.projectionAxis3D);
    else
        userData.MO.processes_{procId}.draw(inputArgs{:},'output', output,...
        options{:},'movieOverlay', movieOverlay);
    end
else
    h = findobj('Tag',graphicTag);
%     if isempty(h) % debugging when non-channel specific tags are
%     mis-matched
%         try 
%            graphicTag = ['^process' num2str(procId) '_'];
%            h = findobj(0,'-regexp','Tag', graphicTag);
%            delete(h);
%         catch
%         end
%     end
    if ~isempty(h), delete(h); end
end

% Get display method and update option status
if isempty(iChan)
    displayMethod = userData.MO.processes_{procId}.displayMethod_{iOutput};
else
    displayMethod = userData.MO.processes_{procId}.displayMethod_{iOutput,iChan}; %% 3D depth specific process???
end
if ~isempty(optFig)
    userData.setOverlayOptions(displayMethod)
end

function deleteViewer()

tags = {'viewerFig','optionsFig','graphFig'};
for i = 1:numel(tags)
    h = findobj(0,'-regexp','Tag', tags{i});
    if ~isempty(h), delete(h); end
end

function updatewellp(src, eventdata, platestack, indw, hpp, mainFig, MO)
        dcolor = get(0, 'DefaultUIControlBackgroundColor');
        siteflags = MO.channels_(1,1).hcsFlags_.siteF;
        hcsview = findobj('Name', 'hcsview');
        userData = get(hcsview, 'UserData'); 
        userData = userData(1);
        if strcmp(userData.lastclick, '0') ~= 1
            lwell = findobj('Tag', userData.lastclick);
            original_color = get(lwell, 'UserData'); % original color saved in each well's UserData
            set(lwell, 'BackgroundColor', original_color);
        end
        if size(indw,2) == 2
            indw = {indw(1), indw(2)};
        end
        pst = platestack{indw{1}, indw{2}};
        cwell = findobj('Tag', MO.channels_(1,1).getGenericName(pst{1,1}));
    % save last clicked well into userdata, in order to be refreshed.
        userData.lastclick = MO.channels_(1,1).getGenericName(pst{1,1});
        set(hcsview, 'UserData', userData);
        current_well_type = get(cwell, 'UserData');
        if sum(current_well_type - [0 1 0]) == 0
            set(cwell, 'BackgroundColor', [0.5 1 0.5]);
        elseif sum(current_well_type - [0 1 1]) == 0
            set(cwell, 'BackgroundColor', [0.8 1 1]);
        else
        set(cwell, 'BackgroundColor', [1 1 1]);
        end
        i4 = 1;
        for i1 = 1:size(siteflags,1)
            for i2 = 1:size(siteflags,2)
                if siteflags(i1, i2) == 0
                    hballsites = uicontrol('Parent', hpp, 'Style', 'pushbutton', 'Position', [i2*22 size(siteflags,1)*22-i1*22+10 22 22],...
                        'enable', 'off');
                else
                    preprocess_flag{1} = 0;
                    sitestr = [MO.channels_(1,1).getGenericName(pst{1,1}),'S',num2str(i4)]; sstr = num2str(i4);
                    hbwp = uicontrol('Parent', hpp, 'Style', 'pushbutton', 'Position', [i2*22 size(siteflags,1)*22-i1*22+10 22 22],...
                        'Tag',['S', num2str(i4)], ...
                        'UserData', dcolor, ...
                        'FontWeight', 'normal', ...
                        'String',sstr,'Callback', {@viewsite, i4, indw, platestack, mainFig, hpp, sitestr, preprocess_flag});%@(hbwp,event) redrawScene(hbwp,guidata(hbwp)));%,...
                    current_well_type = get(cwell, 'UserData');
                    if sum(current_well_type - [0 1 0]) ~= 0 && sum(current_well_type - [0 1 1]) ~= 0
                        if i4 == 1
                            set(hbwp, 'BackgroundColor', [1 1 1]);
                            viewsite(src, eventdata, i4, indw, platestack, mainFig, hpp, sitestr, preprocess_flag);
                        end
                    end
                    %'Callback',{@view4ch, pst, i4,dirurl});
                    if size(indw, 2) >2
                        for isk = 1:length(indw{3})
                            if i4 == indw{3}(isk)
                                preprocess_flag{1} = 1;
                                if size(indw,2)>3
                                    if indw{4} == 1
                                        set(hbwp,'BackgroundColor','cyan');
                                        set(hbwp,'UserData',[0 1 1]);
                                    else
                                        set(hbwp,'BackgroundColor','green');
                                        set(hbwp,'UserData',[0 1 0]);
                                    end
                                    preprocess_flag{2} = indw{5};
                                end
                                set(hbwp, 'Callback', {@viewsite, i4, indw, platestack, mainFig, hpp, sitestr, preprocess_flag});
                            end
                        end
                    end
                    i4 = i4+1;
                end
            end
        end
       
    function viewsite(src, eventdata, inds, indw, platestack, mainFig, hpp, sitestr, preprocess_flag)
        n_site = size(platestack{indw{1},indw{2}},2);
        for id = 1:n_site
            refreshsitestr = findobj(hpp, 'Tag', ['S', num2str(id)]);
            if length(refreshsitestr) > 0
            original_color = get(refreshsitestr(1), 'UserData');
            set(refreshsitestr, 'BackgroundColor', original_color);
            end
        end   
        current_click = findobj('Tag', ['S', num2str(inds)]);
        original_color = get(current_click(1), 'UserData');
        if original_color == [0 1 0] % green
            set(current_click, 'BackgroundColor', [0.5 1 0.5]); % lighter green
        elseif original_color == [0 1 1] % cyan
            set(current_click, 'BackgroundColor', [0.8 1 1]); % lighter cyan
        else
            set(current_click, 'BackgroundColor', [1 1 1]);
        end
        tni = 0;
        for iv = 1:indw{1} %get number of rows
            for ih = 1:size(platestack,2)%get number of columns
                niw = size(platestack{iv,ih},2);%get number of sites within one well
                % doing this loop instead of multiplication can
                % avoid the case of inconsistent number of sites
                % exists across wells.
                if iv == indw{1} && ih == indw{2}
                    tni = tni + inds;
                    break;
                else
                    tni = tni + niw;
                end
            end
        end
        if tni == 0
            tni = 1;
        end
        handles = guihandles(mainFig);
        set(handles.slider_frame, 'Value', tni);
        userdata = get(handles.figure1, 'UserData');
        MD = userdata.MO;
        if MD.isMock()
            load([MD.movieDataPath_ filesep MD.mockMD_.parent.movieDataFileName_]);
            % Duplication of objects in matlab will link all the
            % duplicated objects to be consistent. One solution to it is
            % to have this object < matlab.mixin.copy, which is not
            % desired. Therefore we need to load in the original
            % MovieData if it has been changed during MD_mock
            % loading original MD in this level of callback is to
            % prevent MD_mock been wrapped recursively into itself.
        end
        userdata.MO = MD;
        set(handles.figure1, 'UserData', userdata);
        % Preprocess for mock_MD button
        mockppb = uicontrol('Parent', hpp, 'Style', 'pushbutton', 'Position', [299 80 180 25], ...
            'String', ['Process&View ',sitestr], 'Callback', {@mockMD_preprocess, tni, MD, preprocess_flag});

        redrawScene(handles.figure1, handles);
        %redrawScene(handles,guidata(mainFig));
        
        
        function mockMD_preprocess(src, eventdata, tni, MD, preprocess_flag)
            if MD.isMock()
               load([MD.movieDataPath_ filesep MD.movieDataFileName_]); 
               % Duplication of objects in matlab will link all the
               % duplicated objects to be consistent. One solution to it is
               % to have this object < matlab.mixin.copy, which is not
               % desired. Therefore we need to load in the original
               % MovieData if it has been changed during MD_mock
               % loading original MD in this level of callback is to
               % prevent MD_mock been wrapped recursively into itself.
            end
            if preprocess_flag{1} == 0
            MD_mock = mockMovieData(MD, tni);%tni is the single number index for the site. NOT [col# row# site#]
            MD_mock.sanityCheck;
            else 
                load(preprocess_flag{2}); 
                ds = exist('obj', 'var');
                if ds == 1
                    MD_mock = obj;
                else 
                    dsk = exist('MD', 'var');
                    if dsk == 1
                        MD_mock = MD;
                    end
                end
            end              
            fig = movieSelectorGUI;
            handles = guihandles(fig);
            userData = get(handles.figure1, 'UserData');
            userData.MD = MD_mock;
            pathname = MD_mock.movieDataPath_;
            userData.userDir = pathname;
            M = MD_mock;
            set(handles.figure1, 'UserData', userData);
            
            % Display Movie information
            moviePaths = MD_mock.mockMD_.path;
            nMovies= numel(userData.MD);
            iMovie = get(handles.listbox_movie, 'Value');
            if isempty(userData.MD),
                iMovie=0;
            else
                iMovie=max(1,min(iMovie,nMovies));
            end
            set(handles.listbox_movie,'String',moviePaths,'Value',iMovie);
            set(handles.text_movies, 'String', sprintf('%g/%g movie(s)',iMovie,nMovies))
            
            % Display list information
            listPaths = arrayfun(@getFullPath,userData.ML,'Unif',false);
            nLists= numel(userData.ML);
            iList = get(handles.listbox_movieList, 'Value');
            if isempty(userData.ML),
                iList=0;
            else
                iList=max(1,min(iList,nLists));
            end
            set(handles.listbox_movieList,'String',listPaths);
            set(handles.text_movieList, 'String', sprintf('%g/%g movie list(s)',iList,nLists))
        
function togglecontrol(src, eventdata, MO)
pp = figure('Name','Viewer','Position',[0 0 700 530],...
    'NumberTitle','off','Tag','tog','Toolbar','none','MenuBar','none',...
    'Color',get(0,'defaultUicontrolBackgroundColor'),'Resize','off');
platestack = MO.channels_(1,1).hcsPlatestack_;
wellflag = MO.channels_(1,1).hcsFlags_.wellF;
hp = uipanel('Parent', pp,'Title','Select Wells as Controls','FontSize',10,...
    'Position',[.05 .15 .9 .8]);
pwgc = 0; % plate-well good count
for i2 = 1:size(wellflag,1)
    for i3 = 1:size(wellflag,2)
        
        if i2 == 1
            hth = uicontrol('Parent', hp, 'Style', 'Text', 'Position', [i3*22+22,380, 22,22],...
                'String', num2str(i3));
        end
        if i3 == 1
            htv = uicontrol('Parent', hp, 'Style', 'Text', 'Position', [22,375-i2*22, 22,22],...
                'String', char(i2-1+'A'));
        end
        if wellflag(i2, i3) == 0
            hb = uicontrol('Parent', hp,'Style','pushbutton', 'Position', [i3*22+22,380-i2*22,22,22]...
                ,'Tag','slider_frame',...
                'enable', 'off'); %
        elseif wellflag(i2, i3) == 1
            pwgc = pwgc + 1;
            if pwgc == 1
                ipr1 = i2; ipr2 = i3;
            end
            indexvh = [i2-ipr1+1, i3-ipr2+1];
            wellname = MO.channels_(1,1).getGenericName(platestack{i2-ipr1+1, i3-ipr2+1}{1,1});
            hb(pwgc) = uicontrol('Parent', hp,'Style','togglebutton', 'Position', [i3*22+22,380-i2*22,22,22]...
                ,'TooltipString',[MO.channels_(1,1).getGenericName(platestack{i2-ipr1+1, i3-ipr2+1}{1,1}), ...% has to be named in the same way
                ',',num2str(length(platestack{i2-ipr1+1, i3-ipr2+1})), ...
                ' Sites'],'Tag',['T', wellname],...
                'enable', 'on');%,'Callback', {@mockMDindex, indexvh, platestack, pp});
        end
        
    end
end
             hbend = uicontrol('Parent', pp,'Style','pushbutton', 'Position', [450,20,180,30],...
                 'String', 'Submit as Controls',...
                 'enable', 'on','Callback', {@submittoggles, MO});


% 
% function abs_inx = mockMDindex(src, eventdata, indexvh, platestack, pp)
% in = size(platestack{indexvh(1),indexvh(2)},2);
% for i = 1:in
%     index3 = [indexvh, i];
%     abs_inx(i) = indexing3to1(index3, platestack);
% end
% absi_list = get(pp, 'UserData');
% set(pp, 'UserData', [absi_list abs_inx]);

function submittoggles(src, eventdata, MD)
    pt = findobj('Value', 1, 'Style', 'togglebutton');
    selectedWell = get(pt, 'Tag');
    findemptyh = sum(MD.channels_(1,1).hcsFlags_.wellF);
    findemptyv = sum(MD.channels_(1,1).hcsFlags_.wellF');
    e_c = find(findemptyh == 0);
    e_r = find(findemptyv == 0);
    if length(e_c) == 1
        emptywellc = e_c;
    elseif ~isempty(e_c)
        for i1 = 2:length(e_c)
            if e_c(i1)-e_c(i1-1) ~=1
                emptywellc = e_c(i1-1);
            end
        end
    else
        emptywellc = 0;
    end
    if length(e_r) == 1
        emptywellr = e_r;
    elseif ~isempty(e_r)
        for i3 = 2:length(e_r)
            if e_r(i3)-e_r(i3-1) ~=1
                emptywellr = e_r(i3-1);
            end
        end
    else
        emptywellr = 0;
    end
    tnit = [];
    for i= 1:length(selectedWell)
        wpv = double(selectedWell{i}(2))-64-emptywellc;
        wph = str2double(selectedWell{i}(3:4))-emptywellr;
        for i2 = 1:length(MD.channels_(1,1).hcsPlatestack_{1,1})
           index3(i2,:) = [wpv, wph, i2];
           tni(i2) = indexing3to1(index3(i2,:), MD.channels_(1,1).hcsPlatestack_);
        end
        tnit = [tnit tni];
    end  
    tnit = sort(tnit);
mMD = mockMovieData(MD, tnit);
mMD.sanityCheck;
load([mMD.mockMD_.parent.movieDataPath_ filesep mMD.mockMD_.parent.movieDataFileName_]);
movieViewer(MD, 'refresher', '1');
hf = findobj(0, '-regexp', 'Tag', 'tog');
delete(hf)

function flushcontrol(src, eventdata, MD)
    MD.mMDparent_ = [];   
    rmdir([MD.outputDirectory_ filesep 'controls'], 's');
    parentMDpath = [MD.movieDataPath_ filesep MD.movieDataFileName_];
    save(parentMDpath, 'MD');
    movieViewer(MD);

function tni = indexing3to1(index3, platestack)   
indw = {index3(1),index3(2)};
inds = index3(3);
tni = 0;
for iv = 1:indw{1} %get number of rows
    for ih = 1:size(platestack,2)%get number of columns
        niw = size(platestack{iv,ih},2);%get number of sites within one well
        % doing this loop instead of multiplication can
        % avoid the case of inconsistent number of sites
        % exists across wells.
        if iv == indw{1} && ih == indw{2}
            tni = tni + inds;
            return;
        else
            tni = tni + niw;
        end
    end
end
if tni == 0
    tni = 1;
end

function movieviewerRefresher(MO)
hcsFig = findobj('Name', 'hcsview');
mainFig = findobj('Tag', 'figure1');
mainFig = mainFig(end);
userData = get(hcsFig, 'UserData');
userData.MO = MO;
set(hcsFig, 'UserData', userData);
hpp = findobj('Title','Well Inspector');
if MO.hasMock()
    uicontrol('Parent', hpp, 'Style', 'pushbutton', 'Position', [299 30 180 25], ...
        'String', 'Flush Controls', ...
        'Callback', {@flushcontrol, MO});
end
if MO.hasMock()
    mMDsite_number = [];
    for imMDt = 1:size(MO.mMDparent_,1)
    platestack = MO.channels_(1,1).hcsPlatestack_;
        if size(MO.mMDparent_{imMDt},1) > 1
            for imMD = 1:size(MO.mMDparent_{imMDt},1)
                k1 = MO.mMDparent_{imMDt}(imMD, 1);
                k2 = MO.mMDparent_{imMDt}(imMD,2);
                wellname = MO.channels_(1,1).getGenericName(platestack{k1, k2}{1,1});
                hb = findobj('Tag', wellname);
                set(hb,'BackgroundColor','cyan');
                set(hb,'UserData',[0 1 1]);
                userData.lastclick = wellname;
                set(hcsFig, 'UserData', userData);
                mMDsite_number = [mMDsite_number; MO.mMDparent_{imMDt}(imMD,3)];
                set(hb, 'Callback', {@updatewellp, platestack, {k1, k2, mMDsite_number, 1, MO.mMDparent_{imMDt,2}}, hpp, mainFig, MO});
            end            
        else
            k1 = MO.mMDparent_{imMDt}(1);
            k2 = MO.mMDparent_{imMDt}(2);
            wellname = MO.channels_(1,1).getGenericName(platestack{k1, k2}{1,1});
            hb = findobj('Tag', wellname);
            set(hb,'BackgroundColor','green');
            set(hb,'UserData',[0 1 0]);
            userData.lastclick = wellname;
            set(hcsFig, 'UserData', userData);
            mMDsite_number = MO.mMDparent_{imMDt}(3);
            set(hb, 'Callback', {@updatewellp, platestack, {k1, k2, mMDsite_number, 0, MO.mMDparent_{imMDt,2}}, hpp, mainFig, MO});
        end
    end
end

function out = isBuildDynROIProcessSelected(imageTag, userData)
    if ~isempty(imageTag) && ~strcmp(imageTag,'radiobutton_channels')
        tokens = regexp(imageTag,'process(\d+)','tokens');
        procId=str2double(tokens{1}{1});
        if isa(userData.MO.processes_{procId}, 'BuildDynROIProcess')
            iChan = userData.MO.processes_{procId}.funParams_.ChannelIndex(1);
            s = cached.load(userData.MO.processes_{procId}.outFilePaths_{3, iChan}, '-useCache', true);
            load(s.movieDataDynROICell{1});
            userData.MO = MD; % MD built based on DynROI raw images.
            clear MD;
        end
    end
    out = userData;