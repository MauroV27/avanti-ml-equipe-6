from PIL import Image
import math

def color_bump(value, vmin, base_min):
    if value < vmin:
        return base_min
    return value * 1.5

def change_min(x, y, px, py, comp, bool_operator):
    if bool_operator(comp(x, px), comp(y, py)):
        return {"x": x, "y" : y }
    return {"x": px, "y" : py }

"""
Código que processa a imagem e retorna os parametros extraidos numa lista com os seguintes elementos : 
[img_url, is_r_hand, finger_count, density, angle_x, angle_y, meanX, meanY, mean_middle_point]
"""
def image_preposing(img:Image, img_url:str="", is_r_hand:bool=False, finger_count:int=0):
    color_map = []
    min_color = 92
    img_pixels = img.load()
    mean = {"x": 0, "y": 0}
    p_min_h = {"x": float('inf'), "y": float('inf')}
    p_max_h = {"x": float('-inf'), "y": float('-inf')}
    p_min_v = {"x": float('inf'), "y": float('inf')}
    p_max_v = {"x": float('-inf'), "y": float('-inf')}

    # print("pixels em :", img_pixels[1, 1])

    for y in range(img.height):
        for x in range(img.width):
            original_color = img_pixels[x, y]
            # a = original_color  # Assuming grayscale image
            r = color_bump(original_color, min_color, -1)

            if r > min_color:
                mean['x'] += x
                mean['y'] += y

                p_min_h = change_min(x, y, p_min_h['x'], p_min_h['y'], lambda a, b: a < b, lambda c, d: c or d)
                p_max_h = change_min(p_max_h['x'], p_max_h['y'], x, y, lambda a, b: a > b, lambda c, d: c or d)

                p_min_v = change_min(x, y, p_min_v['x'], p_min_v['y'], lambda a, b: a < b, lambda c, d: c and d)
                p_max_v = change_min(p_max_v['x'], p_max_v['y'], x, y, lambda a, b: a > b, lambda c, d: c and d)

                color_map.append((x, y))

            # output_color = (int(r), int(r), int(r))
            img_pixels[x, y] = int(r) # output_color

    num_points = len(color_map)
    # print("total points ", num_points)

    area = ((p_max_h['x'] - p_min_h['x']) * (p_max_v['y'] - p_min_v['y']))

    mean['x'] /= num_points
    mean['y'] /= num_points

    angle_y = math.atan2(p_max_v['y'] - p_min_v['y'], p_max_v['x'] - p_min_v['x'])
    angle_x = math.atan2(p_max_h['y'] - p_min_h['y'], p_max_h['x'] - p_min_h['x'])

    density = num_points / area

    meanX = (mean['x'] - p_min_h['x']) / (p_max_h['x'] - p_min_h['x'])
    meanY = (mean['y'] - p_min_v['y']) / (p_max_v['y'] - p_min_v['y'])

    # print("typeHand:", img_list[max(img_index - 1, 0)][37:39])
    # print("density:", density)
    # print("angleX:", angle_x)
    # print("angleY:", angle_y)
    # print("meanX:", meanX)
    # print("meanY:", meanY)

    middle_point_in_lines = line_intersection(p_min_h, p_max_h, p_min_v, p_max_v)

    mean_middle_point = math.dist((mean['x'], mean['y']), (middle_point_in_lines['x'], middle_point_in_lines['y']))
    # mean_middle_point = mean_middle_point / ((p_max_h['x'] - p_min_h['x']) * (p_max_v['y'] - p_min_v['y']))
    # print("meanMiddlePoint:", mean_middle_point)

    # img.new('RGB', (128, 128)).save("result.png")
    # img.save("result.png", "PNG")
    # img_pixels.sa

    return [img_url, is_r_hand, finger_count, density, angle_x, angle_y, meanX, meanY, mean_middle_point]

    # scale = WIDTH / 128
    # for point in (ptest, mean, p_min_h, p_max_h, p_min_v, p_max_v):
    #     point['x'] *= scale
    #     point['y'] *= scale

def line_intersection(p_a, p_b, p_c, p_d):
    diviser = ((p_a['x'] - p_b['x']) * (p_c['y'] - p_d['y'])) - ((p_c['x'] - p_d['x']) * (p_a['y'] - p_b['y']))
    value_x = ((p_a['y'] - p_c['y']) * (p_d['x'] - p_c['x'])) - ((p_a['x'] - p_c['x']) * (p_d['y'] - p_c['y']))
    value_y = ((p_a['y'] - p_c['y']) * (p_b['x'] - p_a['x'])) - ((p_a['x'] - p_c['x']) * (p_b['y'] - p_a['y']))

    if diviser == 0:
        return {'collide': value_x == 0 and value_y == 0, 'x': 0, 'y': 0, 't': -2}

    c_x = value_x / diviser
    c_y = value_y / diviser

    if 0 <= c_x <= 1 and 0 <= c_y <= 1:
        return {'collide': True, 'x': p_a['x'] + c_x * (p_b['x'] - p_a['x']), 'y': p_a['y'] + c_x * (p_b['y'] - p_a['y']), 't': c_x}

    return {'collide': False, 'x': 0, 'y': 0, 't': -1}

counter = 0


def image_data_extract(img_url:str, is_r_hand:bool, finger_count:int):

    global counter    
    counter += 1
    if counter % 100 == 0:
        print("Quantidade processada : ", counter)

    img = Image.open(img_url)
    return image_preposing(img, img_url, is_r_hand, finger_count)



if __name__ == "__main__":
# img = Image.open("024db1c5-743a-43b5-8090-1d77877f80cf_2R.png")
    img = Image.open("tsteimg-hand.png").convert("L")
    print(image_preposing(img, "", False, 1))