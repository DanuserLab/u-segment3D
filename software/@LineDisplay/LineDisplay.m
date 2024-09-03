classdef LineDisplay < MovieDataDisplay
    % Concrete display class for displaying points or lines
    % Input data must be expressed in xy coordinate system
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
    
    % Sebastien Besson Jun 2011 (last modified May 2012)
    properties
        Color='r';
        Marker = 'none';
        MarkerSize = 6; 
        LineStyle = '-';
        LineWidth = 1;
        XLabel='';
        YLabel='';
        sfont = {'FontName', 'Helvetica', 'FontSize', 18};
        lfont = {'FontName', 'Helvetica', 'FontSize', 22};
        ButtonDownFcn=[];
    end
    methods
                
        function obj=LineDisplay(varargin)
            obj@MovieDataDisplay(varargin{:})
        end
        function h=initDraw(obj,data,tag,varargin)
            % Plot data and set graphical options
            if(isempty(data))
                h=line([],[],varargin{:});
            else
                h=line(data(:,1),data(:,2),varargin{:});
            end
            set(h,'Tag',tag);
            obj.setLineProperties(h);
            obj.setAxesProperties();
        end
        function updateDraw(obj,h,data)
            % Update handle xData and yData
            if(~isempty(data))
                set(h,'XData',data(:,1),'YData',data(:,2));
            else
                set(h,'XData',[],'YData',[]);
            end
            % obj.setLineProperties(h);
            % obj.setAxesProperties();
        end
        
        function setLineProperties(obj, h)
            set(h, 'MarkerSize', obj.MarkerSize,...
                'Color', obj.Color, 'Marker',obj.Marker,...
                'Linestyle', obj.LineStyle, 'LineWidth', obj.LineWidth,...
                'ButtonDownFcn', obj.ButtonDownFcn);
        end
        
        function setAxesProperties(obj)
            % Set labels and fonts
            if ~isempty(obj.XLabel),xlabel(obj.XLabel,obj.lfont{:}); end
            if ~isempty(obj.YLabel),ylabel(obj.YLabel,obj.lfont{:}); end
            set(gca,'LineWidth', 1.5, obj.sfont{:})
        end
    end    
    
    methods (Static)
        function params=getParamValidators()
            params(1).name='Color';
            params(1).validator=@(x)(ischar(x) || (numel(x)==3 && isnumeric(x)));
            params(2).name='Marker';
            params(2).validator=@ischar;
            params(3).name='LineStyle';
            params(3).validator=@ischar;
            params(4).name='LineWidth';
            params(4).validator=@isscalar;
            params(5).name='XLabel';
            params(5).validator=@ischar;
            params(6).name='YLabel';
            params(6).validator=@ischar;
            params(7).name='sfont';
            params(7).validator=@iscell;
            params(8).name='lfont';
            params(8).validator=@iscell;
            params(9).name='MarkerSize';
            params(9).validator=@isscalar;
            params(10).name='ButtonDownFcn';
            params(10).validator=@(x) isempty(x) || isa(x, 'function_handle');
        end
        function f=getDataValidator()
            f=@isnumeric;
        end
    end    
end