% Values from http://www.olympusfluoview.com/applications/fpcolorpalette.html
% Alexa Fluors: http://www.invitrogen.com/site/us/en/home/References/Molecular-Probes-The-Handbook/Technical-Notes-and-Product-Highlights/The-Alexa-Fluor-Dye-Series.html
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

% Francois Aguet, October 2010

function s = getFluorPropStruct()

s(1).name = 'bfp';
s(1).lambda_em = 440e-9;
s(2).name = 'ebfp';
s(2).lambda_em = 440e-9;
s(3).name = 'cfp';
s(3).lambda_em = 475e-9;
s(4).name = 'egfp';
s(4).lambda_em = 507e-9;
s(5).name = 'gfp';
s(5).lambda_em = 509e-9;
s(6).name = 'alexa488';
s(6).lambda_em = 519e-9;
s(7).name = 'yfp';
s(7).lambda_em = 527e-9;
s(8).name = 'alexa555';
s(8).lambda_em = 565e-9;
s(9).name = 'dtomato';
s(9).lambda_em = 581e-9;
s(10).name = 'tdtomato';
s(10).lambda_em = 581e-9;
s(11).name = 'dsred';
s(11).lambda_em = 583e-9;
s(12).name = 'tagrfp';
s(12).lambda_em = 584e-9;
s(13).name = 'alexa568';
s(13).lambda_em = 603e-9;
s(14).name = 'rfp';
s(14).lambda_em = 607e-9;
s(15).name = 'mrfp';
s(15).lambda_em = 607e-9;
s(16).name = 'mcherry';
s(16).lambda_em = 610e-9;
s(17).name = 'texasred';
s(17).lambda_em = 615e-9;
s(18).name = 'alexa647';
s(18).lambda_em = 665e-9;
s(19).name = 'cy3';
s(19).lambda_em = 570e-9;
s(20).name = 'cy5';
s(20).lambda_em = 670e-9;
s(21).name = 'alexa594';
s(21).lambda_em = 617e-9;
s(22).name = 'dapi';
s(22).lambda_em = 470e-9;
s(23).name = 'fluosphere605';
s(23).lambda_em = 605e-9;
s(24).name = 'meos';
s(24).lambda_em = 563e-9;
s(25).name = 'turborfp';
s(25).lambda_em = 574e-9;
s(26).name = 'mruby2';
s(26).lambda_em = 600e-9;
s(27).name = 'tmr';
s(27).lambda_em = 578e-9;
s(28).name = 'mneongreen';
s(28).lambda_em = 517e-9;
s(29).name = 'cyofp';
s(29).lambda_em = 589e-9;


[~,i] = sort([s.lambda_em]);
s = s(i);