#include "./raylib/src/raylib.h"
#define NN_IMPLEMENTATION_
#include "./nn.h"
#include <string.h>

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 800
#define NEURON_RADIUS 20
#define LAYER_SPACING 200
#define VERTICAL_SPACING 80

#define COST_GRAPH_WIDTH 600
#define COST_GRAPH_HEIGHT 400
#define MAX_COST_HISTORY 1000
#define COST_GRAPH_PADDING 40

// Modern color scheme
#define COLOR_BACKGROUND \
    (Color) { 28, 31, 35, 255 }
#define COLOR_GRID \
    (Color) { 40, 44, 52, 255 }
#define COLOR_TEXT \
    (Color) { 229, 231, 235, 255 }
#define COLOR_ACCENT \
    (Color) { 88, 166, 255, 255 }
#define COLOR_NEURON \
    (Color) { 55, 65, 81, 255 }
#define COLOR_NEURON_BORDER \
    (Color) { 156, 163, 175, 255 }
#define COLOR_COST_LINE \
    (Color) { 249, 115, 22, 255 }

typedef struct
{
    float *cost_history;
    int cost_count;
    float min_cost;
    float max_cost;
} CostVisualizer;

typedef struct
{
    bool running;
    CostVisualizer *cost_vis;
} Visualizer;

Color weight_to_color(float weight)
{
    weight = weight < 0 ? 0 : (weight > 1 ? 1 : weight);
    return (Color){
        (unsigned char)(249 * weight),
        (unsigned char)(115 * (1 - weight)),
        (unsigned char)(22 * (1 - weight)),
        180};
}

Visualizer *init_visualizer()
{
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Neural Network Visualization");
    SetTargetFPS(120);

    Visualizer *vis = malloc(sizeof(Visualizer));
    vis->running = true;

    vis->cost_vis = malloc(sizeof(CostVisualizer));
    vis->cost_vis->cost_history = malloc(sizeof(float) * MAX_COST_HISTORY);
    vis->cost_vis->cost_count = 0;
    vis->cost_vis->min_cost = INFINITY;
    vis->cost_vis->max_cost = -INFINITY;

    return vis;
}

void draw_cost_graph(CostVisualizer *cv, float cost, long long int iterations)
{
    int graph_x = WINDOW_WIDTH - COST_GRAPH_WIDTH - 20;    // 20px padding from right
    int graph_y = (WINDOW_HEIGHT - COST_GRAPH_HEIGHT) / 2; // Centered vertically

    if (cv->cost_count < MAX_COST_HISTORY)
    {
        cv->cost_history[cv->cost_count++] = cost;
    }
    else
    {
        memmove(cv->cost_history, cv->cost_history + 1,
                (MAX_COST_HISTORY - 1) * sizeof(float));
        cv->cost_history[MAX_COST_HISTORY - 1] = cost;
    }

    cv->min_cost = fminf(cost, cv->min_cost);
    cv->max_cost = fmaxf(cost, cv->max_cost);

    DrawRectangle(graph_x, graph_y, COST_GRAPH_WIDTH, COST_GRAPH_HEIGHT, COLOR_BACKGROUND);

    for (int i = 0; i < 10; i++)
    {
        float y = graph_y + COST_GRAPH_PADDING +
                  (i * (COST_GRAPH_HEIGHT - 2 * COST_GRAPH_PADDING) / 9.0f);
        DrawLineEx(
            (Vector2){graph_x + COST_GRAPH_PADDING, y},
            (Vector2){graph_x + COST_GRAPH_WIDTH - COST_GRAPH_PADDING, y},
            1, COLOR_GRID);
    }

    DrawLineEx(
        (Vector2){graph_x + COST_GRAPH_PADDING, graph_y + COST_GRAPH_PADDING},
        (Vector2){graph_x + COST_GRAPH_PADDING, graph_y + COST_GRAPH_HEIGHT - COST_GRAPH_PADDING},
        2, COLOR_TEXT);
    DrawLineEx(
        (Vector2){graph_x + COST_GRAPH_PADDING, graph_y + COST_GRAPH_HEIGHT - COST_GRAPH_PADDING},
        (Vector2){graph_x + COST_GRAPH_WIDTH - COST_GRAPH_PADDING, graph_y + COST_GRAPH_HEIGHT - COST_GRAPH_PADDING},
        2, COLOR_TEXT);

    for (int i = 1; i < cv->cost_count; i++)
    {
        float x1 = graph_x + COST_GRAPH_PADDING +
                   ((i - 1) * (COST_GRAPH_WIDTH - 2 * COST_GRAPH_PADDING)) /
                       (float)MAX_COST_HISTORY;
        float x2 = graph_x + COST_GRAPH_PADDING +
                   (i * (COST_GRAPH_WIDTH - 2 * COST_GRAPH_PADDING)) /
                       (float)MAX_COST_HISTORY;

        float y1 = graph_y + COST_GRAPH_HEIGHT - COST_GRAPH_PADDING -
                   ((cv->cost_history[i - 1] - cv->min_cost) *
                    (COST_GRAPH_HEIGHT - 2 * COST_GRAPH_PADDING)) /
                       (cv->max_cost - cv->min_cost);
        float y2 = graph_y + COST_GRAPH_HEIGHT - COST_GRAPH_PADDING -
                   ((cv->cost_history[i] - cv->min_cost) *
                    (COST_GRAPH_HEIGHT - 2 * COST_GRAPH_PADDING)) /
                       (cv->max_cost - cv->min_cost);

        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, 2, COLOR_COST_LINE);
    }

    char stats_text[64];
    snprintf(stats_text, sizeof(stats_text), "Iteration: %lld", iterations);
    DrawText(stats_text, graph_x + 10, graph_y + 10, 20, COLOR_TEXT);

    snprintf(stats_text, sizeof(stats_text), "Cost: %.6f", cost);
    DrawText(stats_text, graph_x + 10, graph_y + 40, 20, COLOR_TEXT);
}

void draw_network(NN *nn)
{
    int network_width = WINDOW_WIDTH - COST_GRAPH_WIDTH - 40; // 40px total padding
    int total_width = (nn->arch_count - 1) * LAYER_SPACING;
    int start_x = (network_width - total_width) / 2;

    for (size_t layer = 1; layer < nn->arch_count; layer++)
    {
        int x1 = start_x + (layer - 1) * LAYER_SPACING;
        int x2 = start_x + layer * LAYER_SPACING;

        for (size_t i = 0; i < nn->arch[layer - 1]; i++)
        {
            int y1 = WINDOW_HEIGHT / 2 + (i - nn->arch[layer - 1] / 2.0) * VERTICAL_SPACING;

            for (size_t j = 0; j < nn->arch[layer]; j++)
            {
                int y2 = WINDOW_HEIGHT / 2 + (j - nn->arch[layer] / 2.0) * VERTICAL_SPACING;

                float weight = MAT_AT(nn->ws[layer], i, j);
                Color connection_color = weight_to_color(weight);

                DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, 2, connection_color);
            }
        }
    }

    for (size_t layer = 0; layer < nn->arch_count; layer++)
    {
        int x = start_x + layer * LAYER_SPACING;

        for (size_t i = 0; i < nn->arch[layer]; i++)
        {
            int y = WINDOW_HEIGHT / 2 + (i - nn->arch[layer] / 2.0) * VERTICAL_SPACING;

            DrawCircleV((Vector2){x, y}, NEURON_RADIUS, COLOR_NEURON);
            DrawCircleLines(x, y, NEURON_RADIUS, COLOR_NEURON_BORDER);

            if (nn->as[layer] != NULL)
            {
                char activation_text[32];
                snprintf(activation_text, sizeof(activation_text), "%.2f",
                         MAT_AT(nn->as[layer], 0, i));

                int text_width = MeasureText(activation_text, 20);
                DrawText(activation_text, x - text_width / 2, y - 10, 20, COLOR_TEXT);
            }
        }
    }
}

void cleanup_visualizer(Visualizer *vis)
{
    if (vis)
    {
        if (vis->cost_vis)
        {
            free(vis->cost_vis->cost_history);
            free(vis->cost_vis);
        }
        free(vis);
    }
    CloseWindow();
}

void learn_with_visualization(NN *nn, ELEMENT_TYPE eps, ELEMENT_TYPE learning_rate,
                              size_t learning_iterations, Mat *training_input,
                              Mat *training_output, Visualizer *vis)
{
    for (size_t i = 0; i < learning_iterations && vis->running; i++)
    {
        if (WindowShouldClose())
        {
            vis->running = false;
            break;
        }

#if MODEL == DIFF
        size_t arch_count = nn->arch_count;
        ELEMENT_TYPE c = cost_NN(nn, training_input, training_output);
        for (size_t j = 1; j < arch_count; j++)
        {
            diff(nn, training_input, training_output, eps, learning_rate, nn->ws[j], c);
            diff(nn, training_input, training_output, eps, learning_rate, nn->bs[j], c);
        }
#elif MODEL == GRAD_DESC
        delta(nn, training_input, training_output, learning_rate);
#endif

        ELEMENT_TYPE current_cost = cost_NN(nn, training_input, training_output);

        BeginDrawing();
        ClearBackground(COLOR_BACKGROUND);
        draw_network(nn);
        draw_cost_graph(vis->cost_vis, current_cost, i);
        EndDrawing();
    }

    ELEMENT_TYPE cost = cost_NN(nn, training_input, training_output);
    printf("Final Cost: %f\n", cost);
}