function [wellFlags,siteFlags,wavelengthNames,hcsPlatestack]=readIXMHTDFileM(inputPath)

% locate the HTD file
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
htdFileName=dir(fullfile(inputPath, '*.HTD'));
if min(size(htdFileName)) == 0
    wellFlags = 0;
    siteFlags = 0;
    wavelengthNames = 0;
    return;
end
htdFileName=htdFileName(1).name;

% read in HTD file
fid=fopen([inputPath, filesep, htdFileName]);
% skip 4 lines
for i=1:4
    fgetl(fid);
end

% read in number of columns/rows
xWells=readLastNumberFromLineofText(fid);
yWells=readLastNumberFromLineofText(fid);
% read in well flags
wellFlags=false(yWells,xWells);
for y=1:yWells
    tline=fgetl(fid);
    indx=strfind(tline,',');
    for x=1:xWells
        flag=tline(indx(x)+2);
        if strcmp(flag,'T')
            wellFlags(y,x)=true;
        end
    end
end
% skip 1 line
fgetl(fid);
% read in x,y grid dimension for different sites within the same well
xSites=readLastNumberFromLineofText(fid);
ySites=readLastNumberFromLineofText(fid);
% read in site flags
siteFlags=false(ySites,xSites);
for y=1:ySites
    tline=fgetl(fid);
    indx=strfind(tline,',');
    for x=1:xSites
        flag=tline(indx(x)+2);
        if strcmp(flag,'T')
            siteFlags(y,x)=true;
        end
    end
end
% skip 1 line
fgetl(fid);
% read in number of channels
nWavelengths=readLastNumberFromLineofText(fid);
% read in wavelength names
wavelengthNames=cell(nWavelengths,1);
for i=1:nWavelengths
    tline=fgetl(fid);
    indx=strfind(tline,'"');
    wavelengthNames{i}=tline(indx(3)+1:indx(4)-1);
end

hcsPlatestack = plategenerator(wellFlags, siteFlags, wavelengthNames, htdFileName);

end


function n=readLastNumberFromLineofText(fid)

tline=fgetl(fid);
indx=strfind(tline,',');
n=tline(indx+1:end);
n=str2double(n);

end


function hcsPlatestack = plategenerator(wellFlags, siteFlags, wavelengthNames, htdFileName)
ct = 0;
h = waitbar(0, 'Loading HCS Images');
for i4 = 1:length(wavelengthNames)
    for i1 = 1:size(wellFlags,1)
        for i2 = 1:size(wellFlags,2)
            if wellFlags(i1,i2) == 1
                inH = regexp(htdFileName, '.HTD');
                if i2 > 9
                    wellnstr = num2str(i2);
                else
                    wellnstr = strcat('0', num2str(i2));
                end
                for i3 = 1:sum(sum(siteFlags))
                    
                    ct = ct + 1;
                    if ct == 1
                        wpvr = i1; wphr = i2;
                    end

                    hcsPlatestack{i4}{i1-wpvr+1, i2-wphr+1}{i3} = strcat(htdFileName(1:inH-1),'_',char(i1 -1 + 'A'), wellnstr, '_',...
                        's', num2str(i3),'_','w', num2str(i4), '.TIF');
                    
                end
            end
        end
    end
    waitbar(i4/(length(wavelengthNames)));
end
close(h);
end