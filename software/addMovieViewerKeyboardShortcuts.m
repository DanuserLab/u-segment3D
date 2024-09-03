function [ handler ] = addMovieViewerKeyboardShortcuts( varargin )
%addMovieViewerKeyboardShortcuts Adds keyboard shortcuts to movieViewer
% INPUT
% viewerHandle - (optional) Handle to movieViewer Viewer figure
% movieHandle - (optional) Handle to movieViewer Movie figure
%
% OUTPUT
% handler - a function suitable for use with KeyPressFcn
%
% Current Keyboard Shortcuts
% Left / Right Arrows - Reverse / Advance the Frame
% Up / Down Arrows - Raise / Lower the Depth
% Spacebar - Next Frame or Depth, reverses direction at end
% Number keys - Toggle respective channel
% Control with the arrow keys increases the increment from 1 to 5
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

% Mark Kittisopikul
% December 2014

ip = inputParser;
ip.addOptional('viewerHandle',0,@ishandle);
ip.addOptional('movieHandle',0,@ishandle);
ip.parse(varargin{:});

viewerHandle = ip.Results.viewerHandle;
movieHandle = ip.Results.movieHandle;

% find viewer and movie 
if(viewerHandle == 0)
    viewerHandle = findobj(0,'Name','Viewer');
end

if(movieHandle == 0)
    movieHandle = findobj(0,'Name','Movie');
end

% find sliders
slider_frame = findobj(viewerHandle,'Tag','slider_frame');
slider_depth = findobj(viewerHandle,'Tag','slider_depth');

% deal with multiple viewers
if(~isempty(slider_frame))
    slider_frame = slider_frame(end);
end
if(~isempty(slider_depth))
    slider_depth = slider_depth(end);
end

% create boolean flags
enable_frame = ~isempty(slider_frame) && ishandle(slider_frame);
enable_depth = ~isempty(slider_depth) && ishandle(slider_depth);

% disable frame and configure spacebar
reverseDirection = false;
spacebar = { 'rightarrow', 'leftarrow' };
if(enable_frame)
    if(get(slider_frame,'Max') == 1)
        spacebar = { 'downarrow','uparrow'};
        enable_frame = false;
    end
end


handler = @keyPressHandler;

% set handlers
if(ishandle(handle(viewerHandle)))
    set(viewerHandle,'KeyPressFcn',handler)
end

if(ishandle(handle(movieHandle)))
    set(movieHandle,'KeyPressFcn',handler)
end

function keyPressHandler(h,e)

    increment = 1;
    if(~isempty(e.Modifier))
        switch(e.Modifier{1})
            case 'control'
                increment = 5;
        end
    end
    
    %% Spacebar either changes the frame or depth
    switch(e.Key)
        case 'space'
            e = struct();
            e.Key = spacebar{reverseDirection + 1};
    end

    %% Control Time (Frames)
    if(enable_frame)
        minFrame = get(slider_frame,'Min');
        currentFrame = get(slider_frame,'Value');
        maxFrame = get(slider_frame,'Max');
        cb = get(slider_frame,'Callback');
        switch(e.Key)
            case 'leftarrow'
                newFrame = currentFrame -increment;
                reverseDirection = true;
                if(newFrame < minFrame)
                    newFrame = minFrame;
                    reverseDirection = false;
                end

            case 'rightarrow'
                newFrame = currentFrame +increment;
                reverseDirection = false;
                if(newFrame > maxFrame)
                    newFrame = maxFrame;
                    reverseDirection = true;
                end
            otherwise
                newFrame = currentFrame;
        end
        if(newFrame ~= currentFrame)
            set(slider_frame,'Value',newFrame);
            cb(slider_frame,e);
            spacebar = { 'rightarrow', 'leftarrow' };
        end
    end

    %% Control Z (Depth)
    if(enable_depth)
        minDepth = get(slider_depth,'Min');
        currentDepth = get(slider_depth,'Value');
        maxDepth = get(slider_depth,'Max');
        cb = get(slider_depth,'Callback');
        switch(e.Key)
            case 'uparrow'
                newDepth = currentDepth -increment;

                reverseDirection = true;
                if(newDepth < minDepth)
                    newDepth = minDepth;
                    reverseDirection = false;
                end
            case 'downarrow'
                newDepth = currentDepth +increment;
                reverseDirection = false;
                if(newDepth > maxDepth)
                    newDepth = maxDepth;
                    reverseDirection = true;
                end
            otherwise
                newDepth = currentDepth;
        end
        if(newDepth ~= currentDepth)
            set(slider_depth,'Value',newDepth);
            cb(slider_depth,e);
            spacebar = { 'downarrow','uparrow'};
        end
    end
    
    %% Toggle channels using the number keys
    num = str2double(e.Key);
    if(~isnan(num))
        channel_box = findobj(viewerHandle,'Tag',['checkbox_channel' e.Key]);
        if(~isempty(channel_box))
            value = get(channel_box,'Value');
            set(channel_box,'Value',~value);
            cb = get(channel_box,'Callback');
            cb(channel_box,e);
        end
    end
end

end

