"""
3d plotting utilities for aerobench
"""

import math
import time
import os
import traceback

from scipy.io import loadmat

import numpy as np
from numpy import rad2deg

# these imports are needed for 3d plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from aerobench.visualize import plot
from aerobench.util import StateIndex

# res：这是一个包含动画所需数据的对象或对象列表。这个数据通常包含了时间序列、状态（如位置、速度、姿态）、可能的飞行模式等信息。这是动画的主要数据源。
#
# filename：这是输出动画的文件名。根据文件扩展名（如 .mp4 或 .gif），动画将被保存为相应格式的文件。
#
# viewsize：这定义了动画视图在水平方向上的大小。它可以影响动画中展示的场景范围。
#
# viewsize_z：这定义了动画视图在垂直方向上（通常是高度）的大小。
#
# f16_scale：这是飞行器模型的缩放比例。这个参数决定了动画中飞行器模型的大小。
#
# trail_pts：这指定了动画中飞行轨迹的点数。较高的值会生成更长的轨迹线。
#
# elev：这是动画视图的仰角（即在垂直方向上的视角）。
#
# azim：这是动画视图的方位角（即在水平方向上的视角）。
#
# skip_frames：这是一个在动画中跳过特定帧的选项，用于控制动画的速度和平滑度。
#
# chase：这是一个布尔值，指定是否采用“追踪”视角，即摄像机视角是否跟随飞行器移动。
#
# fixed_floor：这是一个布尔值，指定地面是否在动画中保持固定位置，或者随着飞行器移动而移动。
#
# init_extra 和 update_extra：这些是函数或函数列表，用于在动画中添加和更新额外的图形元素。init_extra 在动画开始时被调用来初始化额外元素，而 update_extra 在每一帧中被调用来更新这些元素


def get_script_path(filename=__file__):
    """get the path this script"""
    return os.path.dirname(os.path.realpath(filename))


def make_anim(res, filename, viewsize=1000, viewsize_z=1000, f16_scale=30, trail_pts=60,
              elev=30, azim=45, skip_frames=None, chase=False, fixed_floor=False,
              init_extra=None, update_extra=None):
    """
    make a 3d plot of the GCAS maneuver.

    see examples/anim3d folder for examples on usage
    """

    plot.init_plot()
    start = time.time()

    if not isinstance(res, list):
        res = [res]

    if not isinstance(viewsize, list):
        viewsize = [viewsize]

    if not isinstance(viewsize_z, list):
        viewsize_z = [viewsize_z]

    if not isinstance(f16_scale, list):
        f16_scale = [f16_scale]

    if not isinstance(trail_pts, list):
        trail_pts = [trail_pts]

    if not isinstance(elev, list):
        elev = [elev]

    if not isinstance(azim, list):
        azim = [azim]

    if not isinstance(skip_frames, list):
        skip_frames = [skip_frames]

    if not isinstance(chase, list):
        chase = [chase]

    if not isinstance(fixed_floor, list):
        fixed_floor = [fixed_floor]

    if not isinstance(init_extra, list):
        init_extra = [init_extra]

    if not isinstance(update_extra, list):
        update_extra = [update_extra]

    #####
    # fill in defaults
    if filename == '':
        full_plot = False
    else:
        full_plot = True

    for i, skip in enumerate(skip_frames):
        if skip is not None:#如果skip已经有值，则跳过
            continue

        if filename == '':  # plot to the screen
            skip_frames[i] = 5
        elif filename.endswith('.gif'):
            skip_frames[i] = 2
        else:
            skip_frames[i] = 1  # plot every frame（动图？）

    if filename == '':
        filename = None

    ##
    all_times = []
    all_states = []
    all_modes = []
    all_ps_list = []
    all_Nz_list = []
    for r, skip in zip(res, skip_frames):#skip_frames得和res长度一致，skip作用是跳帧，加快动画进程，但是有可能略过关键细节
        t = r['times']
        s = r['states']
        m = r['modes']
        ps = r['ps_list']
        Nz = r['Nz_list']
        missile_states=r['missile']
        t = t[0::skip]
        s = s[0::skip]
        m = m[0::skip]
        ps = ps[0::skip]
        Nz = Nz[0::skip]
        missile_states=missile_states[0::skip]
        all_times.append(t)
        all_states.append(s)
        all_modes.append(m)
        all_ps_list.append(ps)
        all_Nz_list.append(Nz)
    ##
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ##

    parent = get_script_path()
    plane_point_data = os.path.join(parent, 'f-16.mat')

    data = loadmat(plane_point_data)
    f16_pts = data['V']
    f16_faces = data['F']

    plane_polys = Poly3DCollection([], color=None if full_plot else 'k')
    ax.add_collection3d(plane_polys)

    ax.set_xlabel('X [ft]', fontsize=14)
    ax.set_ylabel('Y [ft]', fontsize=14)
    ax.set_zlabel('Altitude [ft]', fontsize=14)

    # text指定这些text在图表中的位置
    fontsize = 14
    time_text = ax.text2D(0.05, 0.97, "", transform=ax.transAxes, fontsize=fontsize)
    mode_text = ax.text2D(0.95, 0.97, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alt_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
    v_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alpha_text = ax.text2D(0.05, 0.89, "", transform=ax.transAxes, fontsize=fontsize)
    beta_text = ax.text2D(0.95, 0.89, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    nz_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, fontsize=fontsize)
    ps_text = ax.text2D(0.95, 0.85, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    ang_text = ax.text2D(0.5, 0.81, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

    trail_line, = ax.plot([], [], [], color='r', lw=2, zorder=50)
    trail_line2, = ax.plot([], [], [], color='b', lw=2, zorder=50, marker='s', markersize=1, markevery=[-1])


    extra_lines = []

    for func in init_extra:
        if func is not None:
            extra_lines.append(func(ax))
        else:
            extra_lines.append([])

    first_frames = []
    frames = 0

    for t in all_times:#all_times长度就是1，t是一个列表，记录每个时间步的程序运行时间
        first_frames.append(frames)
        frames += len(t)#获取总帧数

    def anim_func(global_frame):#更新动画帧
        'updates for the animation frame'

        index = 0
        first_frame = False

        for i, f in enumerate(first_frames):#循环就进行一次f=frames
            if global_frame >= f:
                index = i

                if global_frame == f:
                    first_frame = True
                    break

        frame = global_frame - first_frames[index]#第二张图片帧数从头开始
        states = all_states[index]
        times = all_times[index]
        modes = all_modes[index]
        Nz_list = all_Nz_list[index]
        ps_list = all_ps_list[index]

        print(f"Frame: {global_frame}/{frames} - Index {index} frame {frame}/{len(times)}")

        speed = states[frame][0]
        alpha = states[frame][1]
        beta = states[frame][2]
        alt = states[frame][11]

        phi = states[frame][StateIndex.PHI]
        theta = states[frame][StateIndex.THETA]
        psi = states[frame][StateIndex.PSI]

        dx = states[frame][StateIndex.POS_E]#东向位移
        dy = states[frame][StateIndex.POS_N]#北向位移
        dz = states[frame][StateIndex.ALT]

        if first_frame:
            ax.view_init(elev[index], azim[index])

            for i, lines in enumerate(extra_lines):
                for line in lines:
                    line.set_visible(i == index)

        time_text.set_text('t = {:.2f} sec'.format(times[frame]))

        # if chase[index]:
        #     ax.view_init(elev[index], rad2deg(-psi) - 90.0)

        colors = ['red', 'blue', 'green', 'magenta']

        mode_names = []

        for mode in modes:
            if not mode in mode_names:
                mode_names.append(mode)

        mode = modes[frame]
        mode_index = modes.index(mode)
        col = colors[mode_index % len(colors)]
        mode_text.set_color(col)
        mode_text.set_text('Mode: {}'.format(mode.capitalize()))

        alt_text.set_text('h = {:.2f} ft'.format(alt))
        v_text.set_text('V = {:.2f} ft/sec'.format(speed))

        alpha_text.set_text('$\\alpha$ = {:.2f} deg'.format(rad2deg(alpha)))
        beta_text.set_text('$\\beta$ = {:.2f} deg'.format(rad2deg(beta)))

        nz_text.set_text('$N_z$ = {:.2f} g'.format(Nz_list[frame]))
        ps_text.set_text('$p_s$ = {:.2f} deg/sec'.format(rad2deg(ps_list[frame])))

        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(\
            rad2deg(phi), rad2deg(theta), rad2deg(psi)))

        s = f16_scale[index]
        s = 25 if s is None else s
        pts = scale3d(f16_pts, [-s, s, s])

        pts = rotate3d(pts, theta, psi - math.pi/2, -phi)

        size = viewsize[index]
        size = 1000 if size is None else size
        minx = dx - size
        maxx = dx + size
        miny = dy - size
        maxy = dy + size

        vz = viewsize_z[index]
        vz = 1000 if vz is None else vz

        if fixed_floor[index]:
            minz = 0
            maxz = vz
        else:
            minz = dz - vz
            maxz = dz + vz

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_zlim([minz, maxz])
        
        # ax.set_xlim([-20000, 20000])
        # ax.set_ylim([-20000, 20000])
        # ax.set_zlim([-20000, 20000])

        verts = []
        fc = []
        ec = []
        count = 0

        # draw ground
        # if minz <= 0 <= maxz:
        #     z = 0
        #     verts.append([(minx, miny, z), (maxx, miny, z), (maxx, maxy, z), (minx, maxy, z)])
        #     fc.append('0.8')
        #     ec.append('0.8')

        # draw f16
        for face in f16_faces:
            face_pts = []

            count = count + 1

            if not full_plot and count % 10 != 0:
                continue

            for findex in face:
                face_pts.append((pts[findex-1][0] + dx, \
                    pts[findex-1][1] + dy, \
                    pts[findex-1][2] + dz))

            verts.append(face_pts)
            fc.append('0.2')
            ec.append('0.2')

        plane_polys.set_verts(verts)
        plane_polys.set_facecolor(fc)
        plane_polys.set_edgecolor(ec)

        # do trail
        t = trail_pts[index]
        t = 200 if t is None else t
        trail_len = t // skip_frames[index]
        start_index = max(0, frame-trail_len)

        pos_xs = [pt[StateIndex.POS_E] for pt in states]
        pos_ys = [pt[StateIndex.POS_N] for pt in states]
        pos_zs = [pt[StateIndex.ALT] for pt in states]
        pos_xs2=[pt[0] for pt in missile_states]
        pos_ys2=[pt[1] for pt in missile_states]
        pos_zs2=[pt[2] for pt in missile_states]
        # print(pos_ys2,'\n',len(pos_ys2))#900
        # print(pos_ys,'\n',len(pos_ys))#60
        trail_line.set_data(np.asarray(pos_xs[start_index:frame]), np.asarray(pos_ys[start_index:frame]))
        trail_line.set_3d_properties(np.asarray(pos_zs[start_index:frame]))
        trail_line2.set_data(np.asarray(pos_xs2[start_index:frame]).ravel(), np.asarray(pos_ys2[start_index:frame]).ravel())
        trail_line2.set_3d_properties(np.asarray(pos_zs2[start_index:frame]).ravel())
    


        if update_extra[index] is not None:
            update_extra[index](frame)

    plt.tight_layout()
    interval = 30

    # if filename.endswith('.gif'):
    #     interval = 60

    anim_obj = animation.FuncAnimation(fig, anim_func, frames, interval=interval, \
        blit=False, repeat=True)

    if filename is not None:

        if filename.endswith('.gif'):
            print("\nSaving animation to '{}' using 'imagemagick'...".format(filename))
            anim_obj.save(filename, dpi=60, writer='imagemagick') # dpi was 80
            print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
        else:
            fps = 40
            codec = 'libx264'

            print("\nSaving '{}' at {:.2f} fps using ffmpeg with codec '{}'.".format(filename, fps, codec))

            # if this fails do: 'sudo apt-get install ffmpeg'
            try:
                extra_args = []

                if codec is not None:
                    extra_args += ['-vcodec', str(codec)]

                anim_obj.save(filename, fps=fps, extra_args=extra_args)
                print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
            except AttributeError:
                traceback.print_exc()
                print("\nSaving video file failed! Is ffmpeg installed? Can you run 'ffmpeg' in the terminal?")
    else:
        plt.show()


def scale3d(pts, scale_list):
    """scale a 3d ndarray of points, and return the new ndarray"""

    assert len(scale_list) == 3

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]

    return rv


def rotate3d(pts, theta, psi, phi):
    """rotates an ndarray of 3d points, returns new list"""

    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    sinPsi = math.sin(psi)
    cosPsi = math.cos(psi)
    sinPhi = math.sin(phi)
    cosPhi = math.cos(phi)

    transform_matrix = np.array([ \
        [cosPsi * cosTheta, -sinPsi * cosTheta, sinTheta], \
        [cosPsi * sinTheta * sinPhi + sinPsi * cosPhi, \
        -sinPsi * sinTheta * sinPhi + cosPsi * cosPhi, \
        -cosTheta * sinPhi], \
        [-cosPsi * sinTheta * cosPhi + sinPsi * sinPhi, \
        sinPsi * sinTheta * cosPhi + cosPsi * sinPhi, \
        cosTheta * cosPhi]], dtype=float)

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        rv[i] = np.dot(pts[i], transform_matrix)

    return rv
